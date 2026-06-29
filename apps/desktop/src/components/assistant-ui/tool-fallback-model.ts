import { type ToolTitleKey, translateNow } from '@/i18n'
import { normalizeExternalUrl } from '@/lib/external-link'
import { summarizeShellCommand } from '@/lib/summarize-command'
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

export interface ToolTitleAction {
  prefix: string
  suffix: string
  text: string
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
  titleAction?: ToolTitleAction
  tone: ToolTone
}

interface ToolMeta {
  done: string
  icon?: string
  pending: string
  pendingAction: string
  tone: ToolTone
}

interface ToolMetaSpec {
  icon?: string
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

const FILE_EDIT_TOOL_NAMES = new Set(['edit_file', 'patch', 'write_file'])

export function isFileEditTool(toolName: string): boolean {
  return FILE_EDIT_TOOL_NAMES.has(toolName)
}

export interface DiffLineStats {
  added: number
  removed: number
}

export function countDiffLineStats(diff: string): DiffLineStats {
  let added = 0
  let removed = 0

  for (const line of diff.split('\n')) {
    if (line.startsWith('+') && !line.startsWith('+++')) {
      added += 1
    } else if (line.startsWith('-') && !line.startsWith('---')) {
      removed += 1
    }
  }

  return { added, removed }
}

function fileEditPath(args: Record<string, unknown>, result: Record<string, unknown>): string {
  return (
    firstStringField(args, ['path', 'file', 'filepath']) ||
    firstStringField(result, ['path', 'file', 'filepath', 'resolved_path']) ||
    htmlPathFromInlineDiff(firstStringField(result, ['inline_diff', 'diff']))
  )
}

function fileEditBasename(path: string): string {
  const normalized = path.replace(/\\/g, '/').trim()

  return normalized.split('/').filter(Boolean).pop() || normalized
}

function numericField(record: Record<string, unknown>, key: string): number | undefined {
  const value = record[key]

  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function readFileLineLabel(args: Record<string, unknown>, result: Record<string, unknown>): string {
  if (numericField(args, 'offset') === undefined && numericField(args, 'limit') === undefined) {
    return ''
  }

  const content = firstStringField(result, ['content'])
  const offset = numericField(args, 'offset')
  const limit = numericField(args, 'limit')

  if (offset !== undefined && offset > 0) {
    if (limit === undefined || limit <= 1) {
      return `L${offset}`
    }

    return `L${offset}-${offset + limit - 1}`
  }

  const lines = content
    .split('\n')
    .map(line => /^(\d+)\|/.exec(line)?.[1])
    .filter((line): line is string => !!line)
    .map(Number)

  if (lines.length === 0) {
    return ''
  }

  const start = lines[0]!
  const end = lines[lines.length - 1]!

  return start === end ? `L${start}` : `L${start}-${end}`
}

function readFileDisplayTarget(args: Record<string, unknown>, result: Record<string, unknown>): string {
  const inherited = firstStringField(args, ['context', 'preview'])

  if (inherited) {
    return inherited
  }

  const path = firstStringField(args, ['path', 'file', 'filepath'])

  if (!path) {
    return ''
  }

  const lineLabel = readFileLineLabel(args, result)

  return [fileEditBasename(path), lineLabel].filter(Boolean).join(' ')
}

const TOOL_META: Record<ToolTitleKey, ToolMetaSpec> = {
  browser_click: {
    icon: 'globe',
    tone: 'browser'
  },
  browser_fill: {
    icon: 'globe',
    tone: 'browser'
  },
  browser_navigate: {
    icon: 'globe',
    tone: 'browser'
  },
  browser_snapshot: {
    icon: 'globe',
    tone: 'browser'
  },
  browser_take_screenshot: {
    icon: 'file-media',
    tone: 'browser'
  },
  browser_type: {
    icon: 'globe',
    tone: 'browser'
  },
  clarify: {
    icon: 'question',
    tone: 'agent'
  },
  cronjob: {
    icon: 'watch',
    tone: 'agent'
  },
  edit_file: { icon: 'edit', tone: 'file' },
  execute_code: {
    icon: 'terminal',
    tone: 'terminal'
  },
  image_generate: {
    icon: 'file-media',
    tone: 'image'
  },
  list_files: {
    icon: 'files',
    tone: 'file'
  },
  patch: { icon: 'edit', tone: 'file' },
  read_file: { icon: 'file', tone: 'file' },
  search_files: {
    icon: 'search',
    tone: 'file'
  },
  session_search_recall: {
    icon: 'search',
    tone: 'agent'
  },
  terminal: {
    icon: 'terminal',
    tone: 'terminal'
  },
  todo: { icon: 'tools', tone: 'agent' },
  vision_analyze: {
    icon: 'eye',
    tone: 'image'
  },
  web_extract: { icon: 'globe', tone: 'web' },
  web_search: { icon: 'search', tone: 'web' },
  write_file: { icon: 'edit', tone: 'file' }
}

function isToolTitleKey(name: string): name is ToolTitleKey {
  return name in TOOL_META
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

const PREFIX_META: { icon?: string; labelKey: string; prefix: string; tone: ToolTone }[] = [
  { prefix: 'browser_', labelKey: 'browser', icon: 'globe', tone: 'browser' },
  { prefix: 'web_', labelKey: 'web', icon: 'globe', tone: 'web' }
]

function toolMeta(name: string): ToolMeta {
  if (isToolTitleKey(name)) {
    const meta = TOOL_META[name]

    return {
      done: translateNow(`assistant.tool.titles.${name}.done`),
      pending: translateNow(`assistant.tool.titles.${name}.pending`),
      pendingAction: translateNow(`assistant.tool.titles.${name}.pendingAction`),
      icon: meta.icon,
      tone: meta.tone
    }
  }

  const action = titleForTool(name)
  const prefix = PREFIX_META.find(p => name.startsWith(p.prefix))

  if (prefix) {
    const prefixLabel = translateNow(`assistant.tool.prefixes.${prefix.labelKey}`)

    return {
      done: translateNow('assistant.tool.titleTemplates.prefixedDone', prefixLabel, action),
      pending: translateNow('assistant.tool.titleTemplates.runningPrefixedTool', prefixLabel, action),
      pendingAction: translateNow('assistant.tool.actions.running'),
      icon: prefix.icon,
      tone: prefix.tone
    }
  }

  return {
    done: action,
    pending: translateNow('assistant.tool.titleTemplates.runningTool', action),
    pendingAction: translateNow('assistant.tool.actions.running'),
    tone: 'default'
  }
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

// Each tool result is server-capped (~100KB), but a turn over a big directory
// stacks many rows; painting/serializing them all floods the renderer (freeze,
// then OOM). Clamp every inline-painted payload to a bounded slice — the row's
// Copy button still reads the uncapped `view.detail` for the full output.
export const MAX_TOOL_RENDER_CHARS = 20_000

export function clampForDisplay(value: string, max = MAX_TOOL_RENDER_CHARS): string {
  if (value.length <= max) {
    return value
  }

  const omitted = value.length - max

  return `${value.slice(0, max)}\n\n… ${omitted.toLocaleString()} more characters truncated — use Copy for the full output.`
}

function prettyJson(value: unknown): string {
  const raw = typeof value === 'string' ? value : JSON.stringify(value, null, 2)

  return clampForDisplay(raw ?? '')
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

  if (isFileEditTool(toolName)) {
    return htmlPathFromInlineDiff(firstStringField(result, ['inline_diff', 'diff']))
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
  const record = parseMaybeObject(result)

  for (const key of ['inline_diff', 'diff']) {
    const value = record[key]

    if (typeof value === 'string' && value.trim()) {
      return stripInlineDiffChrome(value)
    }
  }

  return ''
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

function cronScalar(value: unknown): string {
  if (typeof value === 'string') {
    return value.trim()
  }

  if (typeof value === 'number' && Number.isFinite(value)) {
    return String(value)
  }

  return ''
}

function formatCronTime(iso: string): string {
  const ts = Date.parse(iso)

  if (Number.isNaN(ts)) {
    return iso
  }

  return new Date(ts).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

function cronjobSubtitle(argsRecord: Record<string, unknown>, resultRecord: Record<string, unknown>): string {
  const jobs = Array.isArray(resultRecord.jobs) ? resultRecord.jobs : null

  if (jobs) {
    return jobs.length ? `${jobs.length} cron job${jobs.length === 1 ? '' : 's'}` : 'No cron jobs'
  }

  const message = firstStringField(resultRecord, ['message'])

  if (message) {
    return message
  }

  const action = firstStringField(argsRecord, ['action']) || 'manage'
  const name = firstStringField(resultRecord, ['name']) || firstStringField(argsRecord, ['name', 'job_id'])
  const label = `${action[0]?.toUpperCase() ?? ''}${action.slice(1)}`

  return name ? `${label} ${name}` : `Cron ${action}`
}

function cronjobDetail(argsRecord: Record<string, unknown>, resultRecord: Record<string, unknown>): string {
  const jobs = Array.isArray(resultRecord.jobs) ? resultRecord.jobs : null

  if (jobs) {
    if (!jobs.length) {
      return 'No cron jobs scheduled'
    }

    return jobs
      .slice(0, 20)
      .map(job => {
        const row = isRecord(job) ? job : {}
        const name = firstStringField(row, ['name', 'id']) || 'job'
        const sched = firstStringField(row, ['schedule_display', 'schedule'])

        return sched ? `- ${name} · ${sched}` : `- ${name}`
      })
      .join('\n')
  }

  const nextRun = cronScalar(resultRecord.next_run_at)

  const rows: [string, string][] = [
    ['Schedule', cronScalar(resultRecord.schedule)],
    ['Repeat', cronScalar(resultRecord.repeat)],
    ['Delivery', cronScalar(resultRecord.deliver)],
    ['Next run', nextRun ? formatCronTime(nextRun) : '']
  ]

  const lines = rows.filter(([, value]) => value).map(([key, value]) => `${key}: ${value}`)

  return lines.length ? lines.join('\n') : fallbackDetailText(argsRecord, resultRecord)
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

    const command = firstStringField(argsRecord, ['context', 'preview', 'command', 'code']) || contextValue(argsRecord)

    return command ? '' : 'Executed command'
  }

  if (toolName === 'read_file' || isFileEditTool(toolName)) {
    const isEdit = isFileEditTool(toolName)

    const path = isEdit
      ? fileEditPath(argsRecord, resultRecord)
      : firstStringField(argsRecord, ['path', 'file', 'filepath'])

    if (path) {
      return path
    }

    if (!isEdit) {
      return fallbackDetailText(argsRecord, resultRecord)
    }

    return inlineDiffFromResult(resultRecord) ? 'Changed file' : ''
  }

  if (toolName === 'web_extract') {
    const url =
      firstStringField(argsRecord, ['url']) ||
      firstStringField(resultRecord, ['url']) ||
      findFirstUrl(argsRecord, resultRecord)

    return url ? hostnameOf(url) : 'Fetched webpage'
  }

  if (toolName === 'cronjob') {
    return cronjobSubtitle(argsRecord, resultRecord)
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

  if (part.toolName === 'read_file' && part.result !== undefined) {
    const content = firstStringField(resultRecord, ['content', 'text', 'data', 'body'])

    if (content) {
      return content
    }
  }

  if (isFileEditTool(part.toolName)) {
    if (inlineDiffFromResult(part.result)) {
      return ''
    }

    const summary = firstStringField(resultRecord, ['message', 'summary'])

    if (summary) {
      return summary
    }

    if (fileEditPath(argsRecord, resultRecord)) {
      return ''
    }

    return fallbackDetailText(argsRecord, resultRecord)
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

  if (part.toolName === 'cronjob') {
    return cronjobDetail(argsRecord, resultRecord)
  }

  return fallbackDetailText(argsRecord, resultRecord)
}

export function toolCopyPayload(part: ToolPart, view: ToolView): { label: string; text: string } {
  const copy = {
    command: translateNow('assistant.tool.copyCommand'),
    content: translateNow('assistant.tool.copyContent'),
    file: translateNow('assistant.tool.copyFile'),
    output: translateNow('assistant.tool.copyOutput'),
    path: translateNow('assistant.tool.copyPath'),
    query: translateNow('assistant.tool.copyQuery'),
    results: translateNow('assistant.tool.copyResults'),
    url: translateNow('assistant.tool.copyUrl'),
    generic: translateNow('common.copy')
  }

  const args = parseMaybeObject(part.args)
  const result = parseMaybeObject(part.result)
  const detail = view.detail.trim()
  const hasSubstantialOutput = detail.length > 16

  if (part.toolName === 'terminal' || part.toolName === 'execute_code') {
    if (hasSubstantialOutput) {
      return { label: copy.output, text: detail }
    }

    const command = firstStringField(args, ['command', 'code']) || contextValue(args)

    if (command) {
      return { label: copy.command, text: command }
    }
  }

  if (part.toolName === 'web_extract') {
    if (hasSubstantialOutput) {
      return { label: copy.content, text: detail }
    }

    const url = firstStringField(args, ['url', 'target']) || findFirstUrl(args, result)

    if (url) {
      return { label: copy.url, text: url }
    }
  }

  if (part.toolName === 'browser_navigate') {
    const url = firstStringField(args, ['url', 'target']) || findFirstUrl(args, result)

    if (url) {
      return { label: copy.url, text: url }
    }
  }

  if (part.toolName === 'web_search') {
    if (view.searchHits?.length) {
      const text = view.searchHits.map(hit => [hit.title, hit.url, hit.snippet].filter(Boolean).join('\n')).join('\n\n')

      return { label: copy.results, text }
    }

    const query = firstStringField(args, ['search_term', 'query']) || contextValue(args)

    if (query) {
      return { label: copy.query, text: query }
    }
  }

  if (part.toolName === 'read_file' && part.result !== undefined) {
    if (hasSubstantialOutput) {
      return { label: copy.file, text: detail }
    }

    const path = firstStringField(args, ['path', 'file', 'filepath'])

    if (path) {
      return { label: copy.path, text: path }
    }
  }

  if (isFileEditTool(part.toolName)) {
    if (view.inlineDiff.trim()) {
      return { label: copy.file, text: view.inlineDiff }
    }

    const path = fileEditPath(args, result)

    if (path) {
      return { label: copy.path, text: path }
    }
  }

  if (detail) {
    return { label: copy.output, text: detail }
  }

  return { label: copy.generic, text: view.title }
}

interface ToolTitleParts {
  action?: ToolTitleAction
  title: string
}

function titlePartsFromAction(title: string, action?: string): ToolTitleParts {
  if (!action) {
    return { title }
  }

  const actionStart = title.indexOf(action)

  if (actionStart < 0) {
    return { title }
  }

  return {
    action: {
      prefix: title.slice(0, actionStart),
      suffix: title.slice(actionStart + action.length),
      text: action
    },
    title
  }
}

function dynamicTitle(
  part: ToolPart,
  args: Record<string, unknown>,
  result: Record<string, unknown>,
  fallback: ToolTitleParts
): ToolTitleParts {
  const verb = (gerund: string, past: string) => (part.result === undefined ? gerund : past)

  const titledAction = (action: string, title: string): ToolTitleParts =>
    titlePartsFromAction(title, part.result === undefined ? action : undefined)

  if (part.toolName === 'web_extract') {
    const url = findFirstUrl(args, result)
    const action = verb(translateNow('assistant.tool.actions.reading'), translateNow('assistant.tool.actions.read'))

    return url
      ? titledAction(action, translateNow('assistant.tool.titleTemplates.actionTarget', action, hostnameOf(url)))
      : fallback
  }

  if (part.toolName === 'browser_navigate') {
    const url = findFirstUrl(args, result)

    if (!url) {
      return fallback
    }

    const failed =
      part.isError || result.success === false || result.ok === false || Boolean(firstStringField(result, ['error']))

    if (failed) {
      const failAction = translateNow('assistant.tool.actions.failedToOpen')

      return titledAction(
        failAction,
        translateNow('assistant.tool.titleTemplates.actionTarget', failAction, hostnameOf(url))
      )
    }

    const action = verb(translateNow('assistant.tool.actions.opening'), translateNow('assistant.tool.actions.opened'))

    return titledAction(action, translateNow('assistant.tool.titleTemplates.actionTarget', action, hostnameOf(url)))
  }

  if (part.toolName === 'web_search') {
    const query = firstStringField(args, ['search_term', 'query']) || contextValue(args)

    const action = verb(
      translateNow('assistant.tool.actions.searching'),
      translateNow('assistant.tool.actions.searched')
    )

    return query
      ? titledAction(
          action,
          translateNow('assistant.tool.titleTemplates.actionQuoted', action, compactPreview(query, 48))
        )
      : fallback
  }

  if (part.toolName === 'read_file') {
    const target = readFileDisplayTarget(args, result)
    const action = verb(translateNow('assistant.tool.actions.reading'), translateNow('assistant.tool.actions.read'))

    return target
      ? titledAction(action, translateNow('assistant.tool.titleTemplates.actionTarget', action, target))
      : fallback
  }

  if (part.toolName === 'terminal' || part.toolName === 'execute_code') {
    const command =
      firstStringField(args, ['context', 'preview']) ||
      firstStringField(args, ['command', 'code']) ||
      contextValue(args)

    if (command) {
      const action =
        part.toolName === 'execute_code'
          ? verb(translateNow('assistant.tool.actions.runningCode'), translateNow('assistant.tool.actions.ranCode'))
          : verb(translateNow('assistant.tool.actions.running'), translateNow('assistant.tool.actions.ran'))

      return titledAction(
        action,
        translateNow(
          'assistant.tool.titleTemplates.actionCommand',
          action,
          compactPreview(summarizeShellCommand(command), 160)
        )
      )
    }
  }

  if (isFileEditTool(part.toolName)) {
    const path = fileEditPath(args, result)

    if (path) {
      return { title: fileEditBasename(path) }
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

  const titleParts = dynamicTitle(
    part,
    argsRecord,
    resultRecord,
    titlePartsFromAction(baseTitle, part.result === undefined ? meta.pendingAction : undefined)
  )

  const title = titleParts.title
  const titleEnriched = title !== baseTitle
  const baseSubtitle = error || toolSubtitle(part, argsRecord, resultRecord)

  const keepSubtitleWithTitle =
    part.toolName === 'terminal' ||
    part.toolName === 'execute_code' ||
    (isFileEditTool(part.toolName) && Boolean(baseSubtitle.trim()))

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
    titleAction: titleParts.action,
    tone: meta.tone
  }
}
