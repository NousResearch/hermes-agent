import { normalizeExternalUrl } from '@/lib/external-link'
import { extractToolErrorMessage, formatToolResultSummary } from '@/lib/tool-result-summary'

export type ToolTone = 'agent' | 'browser' | 'default' | 'file' | 'image' | 'terminal' | 'web'
export type ToolStatus = 'error' | 'running' | 'success' | 'warning'

export interface ToolPart {
  args?: unknown
  isError?: boolean
  result?: unknown
  toolCallId?: string
  toolName: string
  type: 'tool-call'
}

export interface SearchResultRow {
  snippet: string
  title: string
  url: string
}

interface CountMetric {
  count: number
  noun: string
}

export interface ToolView {
  countLabel?: string
  detail: string
  detailLabel: string
  durationLabel?: string
  icon?: string
  imageUrl?: string
  inlineDiff: string
  previewTarget?: string
  rawArgs: string
  rawResult: string
  /** Set for tools whose output naturally contains ANSI escape codes
   *  (terminal/execute_code) so the renderer knows to run them through
   *  the ANSI parser instead of printing them as literals. */
  rendersAnsi?: boolean
  searchHits?: SearchResultRow[]
  /** When the backend reports stderr as a separate stream (terminal /
   *  execute_code), the renderer shows it as its own labeled, neutrally
   *  tinted block under stdout — distinct from an error tone. */
  stderr?: string
  /** When set, the renderer uses stdout+stderr as separate sections and
   *  ignores the merged `detail`. */
  stdout?: string
  status: ToolStatus
  subtitle: string
  title: string
  tone: ToolTone
}

interface ToolMeta {
  done: string
  icon?: string
  pending: string
  tone: ToolTone
}

export interface MessageRunningStateSlice {
  message: {
    status?: {
      type?: string
    }
  }
  thread: {
    isRunning: boolean
  }
}

const TOOL_META: Record<string, ToolMeta> = {
  browser_click: { done: 'Clicked page element', pending: 'Clicking page element', icon: 'globe', tone: 'browser' },
  browser_fill: { done: 'Filled form field', pending: 'Filling form field', icon: 'globe', tone: 'browser' },
  browser_navigate: { done: 'Opened page', pending: 'Opening page', icon: 'globe', tone: 'browser' },
  browser_snapshot: {
    done: 'Captured page snapshot',
    pending: 'Capturing page snapshot',
    icon: 'globe',
    tone: 'browser'
  },
  browser_take_screenshot: {
    done: 'Captured screenshot',
    pending: 'Capturing screenshot',
    icon: 'file-media',
    tone: 'browser'
  },
  browser_type: { done: 'Typed on page', pending: 'Typing on page', icon: 'globe', tone: 'browser' },
  clarify: { done: 'Asked a question', pending: 'Asking a question', icon: 'question', tone: 'agent' },
  edit_file: { done: 'Edited file', pending: 'Editing file', icon: 'edit', tone: 'file' },
  execute_code: { done: 'Ran code', pending: 'Running code', icon: 'terminal', tone: 'terminal' },
  image_generate: { done: 'Generated image', pending: 'Generating image', icon: 'file-media', tone: 'image' },
  list_files: { done: 'Listed files', pending: 'Listing files', icon: 'files', tone: 'file' },
  patch: { done: 'Patched file', pending: 'Patching file', icon: 'diff', tone: 'file' },
  read_file: { done: 'Read file', pending: 'Reading file', icon: 'file', tone: 'file' },
  search_files: { done: 'Searched files', pending: 'Searching files', icon: 'search', tone: 'file' },
  session_search_recall: {
    done: 'Searched session history',
    pending: 'Searching session history',
    icon: 'search',
    tone: 'agent'
  },
  terminal: { done: 'Ran command', pending: 'Running command', icon: 'terminal', tone: 'terminal' },
  todo: { done: 'Updated todos', pending: 'Updating todos', icon: 'tools', tone: 'agent' },
  vision_analyze: { done: 'Analyzed image', pending: 'Analyzing image', icon: 'eye', tone: 'image' },
  web_extract: { done: 'Read webpage', pending: 'Reading webpage', icon: 'globe', tone: 'web' },
  web_search: { done: 'Searched web', pending: 'Searching web', icon: 'search', tone: 'web' },
  write_file: { done: 'Edited file', pending: 'Editing file', icon: 'edit', tone: 'file' }
}

const INLINE_CODE_SPLIT_RE = /(`[^`\n]+`)/g
const CITATION_MARKER_RE = /(?<=[\p{L}\p{N})\].,!?:;"'”’])\[(?:\d+(?:\s*,\s*\d+)*)\](?!\()/gu
const BACKTICK_NOISE_RE = /`{3,}/g

export const selectMessageRunning = (state: MessageRunningStateSlice) =>
  state.thread.isRunning && state.message.status?.type === 'running'

function titleForTool(name: string): string {
  const normalized = name.replace(/^browser_/, '').replace(/^web_/, '')

  return (
    normalized
      .split('_')
      .filter(Boolean)
      .map(part => `${part[0]?.toUpperCase() ?? ''}${part.slice(1)}`)
      .join(' ') || name
  )
}

const PREFIX_META: { icon?: string; prefix: string; tone: ToolTone; verb: string }[] = [
  { prefix: 'browser_', verb: 'Browser', icon: 'globe', tone: 'browser' },
  { prefix: 'web_', verb: 'Web', icon: 'globe', tone: 'web' }
]

function toolMeta(name: string): ToolMeta {
  if (TOOL_META[name]) {
    return TOOL_META[name]
  }

  const action = titleForTool(name)
  const prefix = PREFIX_META.find(p => name.startsWith(p.prefix))

  return prefix
    ? {
        done: `${prefix.verb} ${action}`,
        pending: `Running ${prefix.verb.toLowerCase()} ${action.toLowerCase()}`,
        icon: prefix.icon,
        tone: prefix.tone
      }
    : { done: action, pending: `Running ${action.toLowerCase()}`, tone: 'default' }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

export function compactPreview(value: unknown, max = 72): string {
  let raw: unknown

  if (typeof value === 'string') {
    raw = value
  } else {
    raw = parseMaybeObject(value).context
  }

  if (typeof raw !== 'string') {
    if (raw == null) {
      raw = ''
    } else {
      try {
        raw = JSON.stringify(raw)
      } catch {
        raw = String(raw)
      }
    }
  }

  const line = (raw as string).replace(/\s+/g, ' ').trim()

  return line.length > max ? `${line.slice(0, max - 1)}…` : line
}

function contextValue(value: unknown): string {
  const row = parseMaybeObject(value)

  if (typeof row.context === 'string') {
    return row.context
  }

  if (typeof row.preview === 'string') {
    return row.preview
  }

  return typeof value === 'string' ? value : ''
}

function prettyJson(value: unknown): string {
  return typeof value === 'string' ? value : JSON.stringify(value, null, 2)
}

function parseMaybeObject(value: unknown): Record<string, unknown> {
  if (isRecord(value)) {
    return value
  }

  if (typeof value !== 'string' || !value.trim()) {
    return {}
  }

  try {
    const parsed = JSON.parse(value)

    return isRecord(parsed) ? parsed : {}
  } catch {
    return {}
  }
}

function unwrapToolPayload(value: unknown): unknown {
  const record = parseMaybeObject(value)

  for (const key of ['data', 'result', 'output', 'response', 'payload']) {
    const payload = record[key]

    if (payload !== undefined && payload !== null) {
      return payload
    }
  }

  return value
}

function numberValue(value: unknown): null | number {
  const n = typeof value === 'number' ? value : Number(value)

  return Number.isFinite(n) ? n : null
}

function formatDurationSeconds(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return ''
  }

  if (seconds < 1) {
    const ms = Math.max(1, Math.round(seconds * 1000))

    return `${ms}ms`
  }

  if (seconds < 60) {
    return `${seconds.toFixed(seconds >= 10 ? 0 : 1)}s`
  }

  const wholeSeconds = Math.round(seconds)
  const minutes = Math.floor(wholeSeconds / 60)
  const remSeconds = wholeSeconds % 60

  if (minutes < 60) {
    return remSeconds ? `${minutes}m ${remSeconds}s` : `${minutes}m`
  }

  const hours = Math.floor(minutes / 60)
  const remMinutes = minutes % 60

  return remMinutes ? `${hours}h ${remMinutes}m` : `${hours}h`
}

const COUNT_FIELD_KEYS = [
  'count',
  'total',
  'result_count',
  'results_count',
  'num_results',
  'match_count',
  'matches_count',
  'file_count',
  'files_count',
  'item_count',
  'items_count',
  'search_count',
  'searches_count',
  'source_count',
  'sources_count',
  'document_count',
  'documents_count',
  'updated',
  'added',
  'removed',
  'deleted',
  'created',
  'changed',
  'processed',
  'steps'
] as const

const COUNT_ARRAY_KEYS = ['results', 'items', 'matches', 'files', 'documents', 'sources', 'rows'] as const

const COUNT_EXCLUDED_KEYS = new Set(['duration_s', 'exit_code', 'status_code'])

const COUNT_NOUN_BY_FIELD: Partial<Record<(typeof COUNT_FIELD_KEYS)[number], string>> = {
  count: '',
  total: '',
  result_count: 'result',
  results_count: 'result',
  num_results: 'result',
  match_count: 'match',
  matches_count: 'match',
  file_count: 'file',
  files_count: 'file',
  item_count: 'item',
  items_count: 'item',
  search_count: 'search',
  searches_count: 'search',
  source_count: 'source',
  sources_count: 'source',
  document_count: 'document',
  documents_count: 'document',
  updated: 'item',
  added: 'item',
  removed: 'item',
  deleted: 'item',
  created: 'item',
  changed: 'item',
  processed: 'item',
  steps: 'step'
}

const COUNT_NOUN_BY_ARRAY: Record<(typeof COUNT_ARRAY_KEYS)[number], string> = {
  documents: 'document',
  files: 'file',
  items: 'item',
  matches: 'match',
  results: 'result',
  rows: 'row',
  sources: 'source'
}

const DEFAULT_COUNT_NOUN_BY_TOOL: Record<string, string> = {
  browser_snapshot: 'item',
  list_files: 'file',
  search_files: 'result',
  session_search_recall: 'result',
  todo: 'todo',
  web_search: 'result'
}

function countFromUnknown(value: unknown): null | number {
  if (Array.isArray(value)) {
    return value.length > 0 ? value.length : null
  }

  const n = numberValue(value)

  if (n === null || n <= 0) {
    return null
  }

  return Math.round(n)
}

function singularizeNoun(noun: string): string {
  const normalized = noun.trim().toLowerCase()

  if (!normalized) {
    return ''
  }

  if (normalized.endsWith('ies') && normalized.length > 3) {
    return `${normalized.slice(0, -3)}y`
  }

  if (/(xes|zes|ches|shes|sses)$/.test(normalized) && normalized.length > 3) {
    return normalized.slice(0, -2)
  }

  if (normalized.endsWith('s') && normalized.length > 2 && !normalized.endsWith('ss')) {
    return normalized.slice(0, -1)
  }

  return normalized
}

function pluralizeNoun(noun: string, count: number): string {
  if (count === 1) {
    return noun
  }

  if (noun === 'search') {
    return 'searches'
  }

  if (noun.endsWith('y') && noun.length > 1 && !/[aeiou]y$/i.test(noun)) {
    return `${noun.slice(0, -1)}ies`
  }

  if (/(s|x|z|ch|sh)$/i.test(noun)) {
    return `${noun}es`
  }

  return `${noun}s`
}

function formatCountLabel(metric: CountMetric): string {
  return `${metric.count} ${pluralizeNoun(metric.noun, metric.count)}`
}

function countMetric(count: number, noun: string): CountMetric {
  return { count, noun: singularizeNoun(noun) || 'item' }
}

function normalizeMetricForTool(toolName: string, metric: CountMetric): CountMetric {
  if (toolName === 'web_search') {
    return countMetric(metric.count, 'result')
  }

  return metric
}

function fallbackCountNoun(toolName: string): string {
  return DEFAULT_COUNT_NOUN_BY_TOOL[toolName] || 'item'
}

function dynamicCountNounFromKey(key: string, fallbackNoun: string): string {
  const normalized = key.toLowerCase()

  if (normalized === 'count' || normalized === 'total') {
    return fallbackNoun
  }

  const stripped = normalized.replace(/_(count|total)$/i, '').replace(/^num_/, '')

  return singularizeNoun(stripped) || fallbackNoun
}

function countFromRecord(record: Record<string, unknown>, fallbackNoun: string): CountMetric | null {
  for (const key of COUNT_FIELD_KEYS) {
    const value = record[key]
    const count = countFromUnknown(value)

    if (count !== null) {
      return countMetric(count, COUNT_NOUN_BY_FIELD[key] || fallbackNoun)
    }
  }

  for (const key of COUNT_ARRAY_KEYS) {
    const value = record[key]
    const count = countFromUnknown(value)

    if (count !== null) {
      return countMetric(count, COUNT_NOUN_BY_ARRAY[key] || fallbackNoun)
    }
  }

  for (const [key, value] of Object.entries(record)) {
    if (COUNT_EXCLUDED_KEYS.has(key)) {
      continue
    }

    if (!/_count$|_total$/i.test(key)) {
      continue
    }

    const count = countFromUnknown(value)

    if (count !== null) {
      return countMetric(count, dynamicCountNounFromKey(key, fallbackNoun))
    }
  }

  return null
}

function countFromText(value: string, fallbackNoun: string): CountMetric | null {
  const text = value.trim()

  if (!text) {
    return null
  }

  const unitMatch =
    text.match(/\b(\d+)\s+(results?|items?|files?|matches?|documents?|sources?|searches?|steps?|rows?)\b/i) ||
    text.match(/\b(?:did|found|returned|listed|searched|matched|updated|created|deleted|processed)\s+(\d+)\b/i)

  if (unitMatch?.[1]) {
    const n = Number(unitMatch[1])
    const noun = unitMatch[2] ? singularizeNoun(unitMatch[2]) : fallbackNoun

    return Number.isFinite(n) && n > 0 ? countMetric(Math.round(n), noun) : null
  }

  return null
}

function toolResultCount(
  part: ToolPart,
  argsRecord: Record<string, unknown>,
  resultRecord: Record<string, unknown>
): CountMetric | null {
  if (part.result === undefined) {
    return null
  }

  const fallbackNounByTool = fallbackCountNoun(part.toolName)

  if (part.toolName === 'web_search') {
    const hits = collectResultItems(part.result)

    if (hits.length) {
      return countMetric(hits.length, 'result')
    }
  }

  const directCount = countFromRecord(resultRecord, fallbackNounByTool)

  if (directCount !== null) {
    return normalizeMetricForTool(part.toolName, directCount)
  }

  const payload = unwrapToolPayload(part.result)

  if (isRecord(payload)) {
    const payloadCount = countFromRecord(payload, fallbackNounByTool)

    if (payloadCount !== null) {
      return normalizeMetricForTool(part.toolName, payloadCount)
    }
  }

  const summaryText =
    firstStringField(resultRecord, ['summary', 'message', 'detail']) || fallbackDetailText(argsRecord, resultRecord)

  const textMetric = countFromText(summaryText, fallbackNounByTool)

  return textMetric ? normalizeMetricForTool(part.toolName, textMetric) : null
}

function looksLikeUrl(value: string): boolean {
  return /^https?:\/\//i.test(value)
}

function looksLikePath(value: string): boolean {
  return /^file:\/\//i.test(value) || /^(?:\/|\.{1,2}\/|~\/).+/.test(value)
}

export function isPreviewableTarget(target: string): boolean {
  return Boolean(
    target &&
    (/^file:\/\//i.test(target) ||
      /^(?:\/|\.{1,2}\/|~\/).+\.html?$/i.test(target) ||
      /^https?:\/\/(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])/i.test(target))
  )
}

function stableHash(value: string): string {
  let hash = 0

  for (let index = 0; index < value.length; index += 1) {
    hash = Math.imul(31, hash) + value.charCodeAt(index)
  }

  return Math.abs(hash).toString(36)
}

export function toolPartDisclosureId(part: ToolPart): string {
  if (part.toolCallId) {
    return `tool:${part.toolCallId}`
  }

  return `tool:${part.toolName}:${stableHash(JSON.stringify(part.args ?? ''))}`
}

export function toolGroupDisclosureId(parts: ToolPart[]): string {
  return `tool-group:${parts.map(toolPartDisclosureId).join('|')}`
}

const URL_PATTERN = /https?:\/\/[^\s'"<>)\]]+/i

function findFirstUrl(...sources: unknown[]): string {
  for (const src of sources) {
    if (typeof src === 'string') {
      const m = src.match(URL_PATTERN)

      if (m) {
        return m[0]
      }
    } else if (src && typeof src === 'object') {
      for (const v of Object.values(src as Record<string, unknown>)) {
        const found = findFirstUrl(v)

        if (found) {
          return found
        }
      }
    }
  }

  return ''
}

function hostnameOf(value: string): string {
  try {
    const url = new URL(value)

    return `${url.hostname}${url.pathname && url.pathname !== '/' ? url.pathname : ''}`
  } catch {
    return value
  }
}

export function looksRedundant(title: string, detail: string): boolean {
  if (!detail) {
    return true
  }

  const norm = (input: string) => input.toLowerCase().replace(/\s+/g, ' ').trim()

  return norm(title) === norm(detail)
}

export function cleanVisibleText(text: string): string {
  return text
    .split(INLINE_CODE_SPLIT_RE)
    .map(part =>
      part.startsWith('`')
        ? part
        : part
            .replace(BACKTICK_NOISE_RE, '')
            .replace(CITATION_MARKER_RE, '')
            .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_match, label: string, href: string) => {
              const normalized = normalizeExternalUrl(href)

              return `${label} ${normalized}`
            })
    )
    .join('')
}

function summarizeBrowserSnapshot(snapshot: string): string {
  const count = (re: RegExp) => snapshot.match(re)?.length ?? 0

  const stats = [
    `${count(/button\s+"[^"]+"/g)} buttons`,
    `${count(/link\s+"[^"]+"/g)} links`,
    `${count(/(?:textbox|combobox|searchbox)\s+"[^"]+"/g)} inputs`
  ].join(' · ')

  const labels = Array.from(snapshot.matchAll(/(?:button|link|combobox|textbox)\s+"([^"]+)"/g))
    .map(m => m[1].trim())
    .filter(Boolean)
    .slice(0, 4)

  return labels.length ? `${stats}\nTop controls: ${labels.join(', ')}` : stats
}

function firstStringField(record: Record<string, unknown>, keys: readonly string[]): string {
  for (const key of keys) {
    const value = record[key]

    if (typeof value === 'string' && value.trim()) {
      return value.trim()
    }
  }

  return ''
}

function collectResultItems(value: unknown): unknown[] {
  if (Array.isArray(value)) {
    return value
  }

  const record = parseMaybeObject(value)

  for (const key of [
    'web',
    'results',
    'search_results',
    'sources',
    'web_sources',
    'items',
    'organic_results',
    'organic',
    'matches',
    'documents'
  ]) {
    const candidate = record[key]

    if (Array.isArray(candidate)) {
      return candidate
    }

    if (isRecord(candidate)) {
      const nested = collectResultItems(candidate)

      if (nested.length) {
        return nested
      }
    }
  }

  const payload = unwrapToolPayload(record)

  return payload === record ? [] : collectResultItems(payload)
}

function extractSearchResults(result: unknown, limit = 6): SearchResultRow[] {
  const list = collectResultItems(result)

  return list
    .map(item => {
      const r = parseMaybeObject(item)

      return {
        title: cleanVisibleText(firstStringField(r, ['title', 'name'])),
        url: firstStringField(r, ['url', 'href', 'link']),
        snippet: cleanVisibleText(firstStringField(r, ['snippet', 'description', 'body']))
      }
    })
    .filter(hit => hit.title || hit.url)
    .slice(0, limit)
}

function toolErrorText(part: ToolPart, result: Record<string, unknown>): string {
  const extractedError = extractToolErrorMessage(part.result)

  if (part.isError) {
    return extractedError || (typeof part.result === 'string' && part.result.trim()) || 'Tool returned an error.'
  }

  if (typeof result.error === 'string' && result.error.trim()) {
    return result.error.trim()
  }

  if (extractedError) {
    return extractedError
  }

  if (result.success === false || result.ok === false) {
    return firstStringField(result, ['message', 'reason', 'detail']) || 'Tool returned success=false.'
  }

  if (typeof result.status === 'string' && /\b(error|failed|failure)\b/i.test(result.status)) {
    return firstStringField(result, ['message', 'reason', 'detail']) || `Tool returned status "${result.status}".`
  }

  // A non-zero exit code alone is a weak failure signal: grep returns 1 on
  // no-match, diff returns 1 on differences, piped commands surface the last
  // stage's code, etc. — all routinely produce useful output and aren't
  // failures. Only treat it as an error when the command produced no real
  // output to show; otherwise render the output normally (not red).
  const exit = numberValue(result.exit_code)

  if (exit !== null && exit !== 0) {
    const hasOutput = Boolean(firstStringField(result, ['output', 'stdout', 'stderr'])?.trim())

    return hasOutput ? '' : `Command failed with exit code ${exit}.`
  }

  return ''
}

function toolStatus(part: ToolPart, resultRecord: Record<string, unknown>): ToolStatus {
  if (part.result === undefined) {
    return 'running'
  }

  return toolErrorText(part, resultRecord) ? 'error' : 'success'
}

function durationLabel(resultRecord: Record<string, unknown>): string | undefined {
  const seconds = numberValue(resultRecord.duration_s)

  if (seconds === null || seconds < 0) {
    return undefined
  }

  return formatDurationSeconds(seconds)
}

function toolPreviewTarget(toolName: string, args: Record<string, unknown>, result: Record<string, unknown>): string {
  const direct =
    firstStringField(result, ['preview', 'url', 'target']) ||
    firstStringField(args, ['preview', 'url', 'target', 'path', 'file', 'filepath']) ||
    firstStringField(result, ['path', 'file', 'filepath'])

  if (direct && (looksLikeUrl(direct) || looksLikePath(direct))) {
    return direct
  }

  if (toolName === 'browser_navigate' || toolName === 'web_extract' || toolName === 'web_search') {
    const explicit = firstStringField(args, ['url', 'search_term', 'query']) || firstStringField(result, ['url'])

    return looksLikeUrl(explicit) ? explicit : findFirstUrl(args, result)
  }

  if (toolName === 'write_file' || toolName === 'edit_file') {
    return htmlPathFromInlineDiff(firstStringField(result, ['inline_diff']))
  }

  return ''
}

function toolImageUrl(args: Record<string, unknown>, result: Record<string, unknown>): string {
  const candidate =
    firstStringField(result, ['image_url', 'url', 'path', 'image_path']) ||
    firstStringField(args, ['image_url', 'url', 'path'])

  if (!candidate) {
    return ''
  }

  // Only inline-render images the renderer can actually fetch: data URLs or
  // remote http(s). A bare filesystem path (e.g. vision_analyze's input image)
  // resolves against the dev-server origin and 404s — fall back to the tool's
  // codicon instead of a broken <img>.
  const isDataImage = candidate.toLowerCase().startsWith('data:image/')
  const isRemoteImage = /^https?:\/\//i.test(candidate) && /\.(png|jpe?g|gif|webp|bmp|svg)(\?|#|$)/i.test(candidate)

  return isDataImage || isRemoteImage ? candidate : ''
}

function stripAnsi(value: string): string {
  return value.replace(new RegExp(`${String.fromCharCode(27)}\\[[0-9;]*m`, 'g'), '')
}

export function stripInlineDiffChrome(value: string): string {
  return value
    ? stripAnsi(value)
        .replace(/^\s*┊\s*review diff\s*\n/i, '')
        .trim()
    : ''
}

function htmlPathFromInlineDiff(value: string): string {
  const cleaned = stripInlineDiffChrome(value)

  for (const match of cleaned.matchAll(/(?:^|\s)(?:[ab]\/)?([^\s]+\.html?)(?=\s|$)/gi)) {
    const candidate = match[1]?.trim()

    if (candidate) {
      return candidate
    }
  }

  return ''
}

function stripDividerLines(value: string): string {
  return value
    .split('\n')
    .filter(line => !/^[-=]{3,}\s*$/.test(line.trim()))
    .join('\n')
    .trim()
}

export function inlineDiffFromResult(result: unknown): string {
  const value = parseMaybeObject(result).inline_diff

  return typeof value === 'string' ? stripInlineDiffChrome(value) : ''
}

// Falls back to a string only when there's something concrete to render —
// counts of opaque items/fields are noise, not signal.
function minimalValueSummary(value: unknown): string {
  if (value == null) {
    return ''
  }

  if (typeof value === 'string') {
    return value
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }

  return ''
}

function fallbackDetailText(args: unknown, result: unknown): string {
  const argContext = contextValue(args)
  const resultContext = contextValue(result)

  if (resultContext && resultContext !== argContext) {
    return resultContext
  }

  if (argContext) {
    return argContext
  }

  if (result !== undefined) {
    return formatToolResultSummary(result) || minimalValueSummary(result)
  }

  return formatToolResultSummary(args) || minimalValueSummary(args)
}

function toolSubtitle(
  part: ToolPart,
  argsRecord: Record<string, unknown>,
  resultRecord: Record<string, unknown>
): string {
  const toolName = part.toolName

  if (toolName === 'browser_navigate') {
    const url =
      firstStringField(argsRecord, ['url', 'target']) ||
      firstStringField(resultRecord, ['url']) ||
      findFirstUrl(argsRecord, resultRecord)

    return url ? hostnameOf(url) : 'Navigated in browser'
  }

  if (toolName === 'browser_snapshot') {
    const snapshot = firstStringField(resultRecord, ['snapshot'])

    return snapshot ? summarizeBrowserSnapshot(snapshot) : 'Captured a browser accessibility snapshot'
  }

  if (toolName === 'browser_click') {
    const clicked = firstStringField(resultRecord, ['clicked']) || firstStringField(argsRecord, ['ref', 'target'])

    if (!clicked) {
      return 'Clicked on page'
    }

    return clicked.startsWith('@') ? `Clicked page element (internal ref ${clicked})` : `Clicked ${clicked}`
  }

  if (toolName === 'browser_fill' || toolName === 'browser_type') {
    const field = firstStringField(argsRecord, ['label', 'field', 'ref', 'target'])
    const value = firstStringField(argsRecord, ['value', 'text'])

    return (
      [field && `Field: ${field}`, value && `Value: ${compactPreview(value, 42)}`].filter(Boolean).join(' · ') ||
      'Filled page input'
    )
  }

  if (toolName === 'web_search') {
    const query = firstStringField(argsRecord, ['search_term', 'query']) || contextValue(argsRecord)

    return query ? `Query: ${query}` : 'Queried web sources'
  }

  if (toolName === 'terminal' || toolName === 'execute_code') {
    const output = firstStringField(resultRecord, ['output', 'stdout', 'stderr'])

    const lines = Array.isArray(resultRecord.lines)
      ? resultRecord.lines.filter((line): line is string => typeof line === 'string').join('\n')
      : ''

    const previewSource = (output || lines).trim()

    if (previewSource) {
      const firstMeaningfulLine = previewSource
        .split('\n')
        .map(line => line.trim())
        .find(line => line.length > 0)

      if (firstMeaningfulLine) {
        return compactPreview(firstMeaningfulLine, 160)
      }
    }

    const command = firstStringField(argsRecord, ['command', 'code']) || contextValue(argsRecord)

    return command ? compactPreview(command, 120) : 'Executed command'
  }

  if (toolName === 'read_file' || toolName === 'write_file' || toolName === 'edit_file') {
    const path =
      firstStringField(argsRecord, ['path', 'file', 'filepath']) ||
      htmlPathFromInlineDiff(firstStringField(resultRecord, ['inline_diff']))

    return (
      path ||
      (firstStringField(resultRecord, ['inline_diff']) ? 'Changed file' : fallbackDetailText(argsRecord, resultRecord))
    )
  }

  if (toolName === 'web_extract') {
    const url =
      firstStringField(argsRecord, ['url']) ||
      firstStringField(resultRecord, ['url']) ||
      findFirstUrl(argsRecord, resultRecord)

    return url ? hostnameOf(url) : 'Fetched webpage'
  }

  return (
    compactPreview(formatToolResultSummary(part.result), 120) ||
    compactPreview(resultRecord, 120) ||
    compactPreview(argsRecord, 120) ||
    fallbackDetailText(argsRecord, resultRecord)
  )
}

function toolDetailLabel(toolName: string): string {
  if (toolName === 'web_search') {
    return 'Details'
  }

  if (toolName === 'browser_snapshot') {
    return 'Snapshot summary'
  }

  if (toolName === 'terminal' || toolName === 'execute_code') {
    return 'Command output'
  }

  return ''
}

function toolDetailText(
  part: ToolPart,
  argsRecord: Record<string, unknown>,
  resultRecord: Record<string, unknown>
): string {
  if (part.toolName === 'browser_snapshot') {
    const snapshot = firstStringField(resultRecord, ['snapshot'])

    return snapshot ? summarizeBrowserSnapshot(snapshot) : fallbackDetailText(argsRecord, resultRecord)
  }

  if (part.toolName === 'terminal' || part.toolName === 'execute_code') {
    // Streams are split out into ToolView.stdout / ToolView.stderr by
    // buildToolView so the renderer can label them separately. The merged
    // fallback here is only used when the backend doesn't expose either
    // stream individually.
    const output = firstStringField(resultRecord, ['output', 'stdout', 'stderr'])

    const lines = Array.isArray(resultRecord.lines)
      ? resultRecord.lines.filter((line): line is string => typeof line === 'string').join('\n')
      : ''

    if (output || lines) {
      return [output, lines].filter(Boolean).join('\n')
    }
  }

  if (part.toolName === 'web_extract') {
    const direct = firstStringField(resultRecord, ['content', 'text', 'markdown', 'body', 'summary', 'message'])

    if (direct) {
      return direct.replace(/\s*in\s+\d+(?:\.\d+)?s\s*$/i, '').trim()
    }

    const results = Array.isArray(resultRecord.results) ? resultRecord.results : []

    const aggregated = results
      .map(item => {
        const row = parseMaybeObject(item)

        return firstStringField(row, ['content', 'text', 'markdown', 'body'])
      })
      .filter(Boolean)
      .join('\n\n---\n\n')

    if (aggregated) {
      return aggregated
    }
  }

  if (part.toolName === 'read_file') {
    const content = firstStringField(resultRecord, ['content', 'text', 'data', 'body'])

    if (content) {
      return content
    }
  }

  if (part.toolName === 'write_file' || part.toolName === 'edit_file') {
    return inlineDiffFromResult(part.result) ? '' : fallbackDetailText(argsRecord, resultRecord)
  }

  if (part.toolName === 'web_search') {
    const detail = fallbackDetailText(argsRecord, resultRecord)
    const seconds = numberValue(resultRecord.duration_s)
    const duration = seconds === null ? '' : formatDurationSeconds(seconds)

    if (!duration) {
      return detail
    }

    return detail
      .replace(/^\s*-\s*Duration\s+S\s*:\s*[-+]?[\d.]+(?:e[-+]?\d+)?\s*$/gim, `- Duration: ${duration}`)
      .replace(/\bDuration\s+S\s*:/gi, 'Duration:')
  }

  return fallbackDetailText(argsRecord, resultRecord)
}

export function toolCopyPayload(part: ToolPart, view: ToolView): { label: string; text: string } {
  const args = parseMaybeObject(part.args)
  const result = parseMaybeObject(part.result)
  const detail = view.detail.trim()
  const hasSubstantialOutput = detail.length > 16

  if (part.toolName === 'terminal' || part.toolName === 'execute_code') {
    if (hasSubstantialOutput) {
      return { label: 'Copy output', text: detail }
    }

    const command = firstStringField(args, ['command', 'code']) || contextValue(args)

    if (command) {
      return { label: 'Copy command', text: command }
    }
  }

  if (part.toolName === 'web_extract') {
    if (hasSubstantialOutput) {
      return { label: 'Copy content', text: detail }
    }

    const url = firstStringField(args, ['url', 'target']) || findFirstUrl(args, result)

    if (url) {
      return { label: 'Copy URL', text: url }
    }
  }

  if (part.toolName === 'browser_navigate') {
    const url = firstStringField(args, ['url', 'target']) || findFirstUrl(args, result)

    if (url) {
      return { label: 'Copy URL', text: url }
    }
  }

  if (part.toolName === 'web_search') {
    if (view.searchHits?.length) {
      const text = view.searchHits.map(hit => [hit.title, hit.url, hit.snippet].filter(Boolean).join('\n')).join('\n\n')

      return { label: 'Copy results', text }
    }

    const query = firstStringField(args, ['search_term', 'query']) || contextValue(args)

    if (query) {
      return { label: 'Copy query', text: query }
    }
  }

  if (part.toolName === 'read_file') {
    if (hasSubstantialOutput) {
      return { label: 'Copy file', text: detail }
    }

    const path = firstStringField(args, ['path', 'file', 'filepath'])

    if (path) {
      return { label: 'Copy path', text: path }
    }
  }

  if (part.toolName === 'write_file' || part.toolName === 'edit_file') {
    const path = firstStringField(args, ['path', 'file', 'filepath'])

    if (path) {
      return { label: 'Copy path', text: path }
    }
  }

  if (detail) {
    return { label: 'Copy output', text: detail }
  }

  return { label: 'Copy', text: view.title }
}

function dynamicTitle(
  part: ToolPart,
  args: Record<string, unknown>,
  result: Record<string, unknown>,
  fallback: string
): string {
  const verb = (gerund: string, past: string) => (part.result === undefined ? gerund : past)

  if (part.toolName === 'web_extract') {
    const url = findFirstUrl(args, result)

    return url ? `${verb('Reading', 'Read')} ${hostnameOf(url)}` : fallback
  }

  if (part.toolName === 'browser_navigate') {
    const url = findFirstUrl(args, result)

    return url ? `${verb('Opening', 'Opened')} ${hostnameOf(url)}` : fallback
  }

  if (part.toolName === 'web_search') {
    const query = firstStringField(args, ['search_term', 'query']) || contextValue(args)

    return query ? `${verb('Searching', 'Searched')} “${compactPreview(query, 48)}”` : fallback
  }

  if (part.toolName === 'terminal' || part.toolName === 'execute_code') {
    const command = firstStringField(args, ['command', 'code']) || contextValue(args)

    if (command) {
      const verbText = part.toolName === 'execute_code' ? verb('Running code', 'Ran code') : verb('Running', 'Ran')

      return `${verbText} · ${compactPreview(command, 160)}`
    }
  }

  return fallback
}

export function buildToolView(part: ToolPart, inlineDiff: string): ToolView {
  const argsRecord = parseMaybeObject(part.args)
  const resultRecord = parseMaybeObject(part.result)
  const meta = toolMeta(part.toolName)
  const status = toolStatus(part, resultRecord)
  const error = toolErrorText(part, resultRecord)
  const baseTitle = part.result === undefined ? meta.pending : meta.done
  const title = dynamicTitle(part, argsRecord, resultRecord, baseTitle)
  const titleEnriched = title !== baseTitle
  const baseSubtitle = error || toolSubtitle(part, argsRecord, resultRecord)
  const keepSubtitleWithTitle = part.toolName === 'terminal' || part.toolName === 'execute_code'
  const subtitle = titleEnriched && !error && !keepSubtitleWithTitle ? '' : baseSubtitle
  const detailBody = stripDividerLines(toolDetailText(part, argsRecord, resultRecord))

  const detail = error
    ? [error, detailBody]
        .filter(Boolean)
        .filter((value, index, list) => list.findIndex(entry => entry.trim() === value.trim()) === index)
        .join('\n\n')
    : detailBody

  const searchHits =
    part.toolName === 'web_search' && status !== 'error' ? extractSearchResults(part.result) : undefined

  const resultCount = status === 'error' ? null : toolResultCount(part, argsRecord, resultRecord)

  // For shell/code tools we surface stdout and stderr as separate labeled
  // streams in the renderer. Many CLIs use stderr for informational
  // messages (npm progress, git hints), so we deliberately don't paint
  // stderr destructively even though it's tagged.
  const rendersAnsi = part.toolName === 'terminal' || part.toolName === 'execute_code'
  const stdout = rendersAnsi ? firstStringField(resultRecord, ['stdout']) : ''
  const stderrRaw = rendersAnsi ? firstStringField(resultRecord, ['stderr']) : ''
  // Only attach stderr when the backend actually returned it as its own
  // field — otherwise the merged `detail` already covers it and double-
  // rendering would duplicate output.
  const hasSplitStreams = rendersAnsi && (Boolean(stdout) || Boolean(stderrRaw))

  return {
    countLabel: resultCount ? formatCountLabel(resultCount) : undefined,
    detail,
    detailLabel: error ? 'Error details' : toolDetailLabel(part.toolName),
    durationLabel: durationLabel(resultRecord),
    icon: meta.icon,
    imageUrl: toolImageUrl(argsRecord, resultRecord),
    inlineDiff,
    previewTarget: toolPreviewTarget(part.toolName, argsRecord, resultRecord),
    rawArgs: prettyJson(part.args),
    rawResult: prettyJson(part.result),
    rendersAnsi: rendersAnsi || undefined,
    searchHits: searchHits?.length ? searchHits : undefined,
    stderr: hasSplitStreams ? stderrRaw || undefined : undefined,
    stdout: hasSplitStreams ? stdout || undefined : undefined,
    status,
    subtitle,
    title,
    tone: meta.tone
  }
}
