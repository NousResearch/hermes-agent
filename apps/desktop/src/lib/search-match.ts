/**
 * Shared multi-field search ranking for Desktop lists (command palette, skills,
 * session metadata). Ranking happens in React — cmdk stays keyboard-only.
 *
 * Features:
 * - Multi-field weighted scoring + match-field reporting
 * - Grades: exact > prefix > whole word > word prefix > substring > fuzzy
 * - Query syntax: whitespace = AND; `OR` / `|` between alternatives; quoted
 *   phrases; trailing `*` keeps prefix-only for that term
 * - Light edit-distance fuzzy (≤1–2) for short lists / long-enough terms
 */

import { asText, normalize } from '@/lib/text'

export type SearchField =
  | 'title'
  | 'name'
  | 'label'
  | 'preview'
  | 'description'
  | 'category'
  | 'cwd'
  | 'branch'
  | 'id'
  | 'source'
  | 'keywords'
  | 'body'
  | 'tool'

export type MatchKind = 'exact' | 'prefix' | 'word' | 'word-prefix' | 'substring' | 'fuzzy' | 'keyword'

export interface FieldMatch {
  field: SearchField
  kind: MatchKind
  score: number
  value: string
  /** Inclusive-exclusive UTF-16 code unit ranges into `value`. */
  ranges: Array<[number, number]>
}

export interface RankedHit<T> {
  item: T
  score: number
  matches: FieldMatch[]
}

export interface FieldSpec {
  field: SearchField
  value: unknown
  /** Relative weight; default 1. Keywords default lower via field. */
  weight?: number
}

export interface RankOptions {
  /** Enable edit-distance fuzzy (default true for small lists). */
  fuzzy?: boolean
  /** Soft cap for enabling fuzzy when not set explicitly (default 2000). */
  fuzzyMaxItems?: number
  /** Candidate count — used with fuzzyMaxItems. */
  itemCount?: number
}

const DEFAULT_FIELD_WEIGHT: Partial<Record<SearchField, number>> = {
  title: 1,
  name: 1,
  label: 1,
  preview: 0.85,
  description: 0.85,
  category: 0.7,
  branch: 0.75,
  cwd: 0.65,
  id: 0.55,
  source: 0.5,
  keywords: 0.4,
  body: 0.7,
  tool: 0.8
}

const GRADE: Record<MatchKind, number> = {
  exact: 1,
  prefix: 0.9,
  word: 0.85,
  'word-prefix': 0.8,
  substring: 0.7,
  fuzzy: 0.55,
  keyword: 0.4
}

/** FTS5 snippet markers from the backend. */
export const FTS_MARK_OPEN = '>>>'
export const FTS_MARK_CLOSE = '<<<'

/** Client highlight markers when we re-wrap ranges. */
export const HL_OPEN = '[[m]]'
export const HL_CLOSE = '[[/m]]'

// ── Query parsing ──────────────────────────────────────────────────────────

export interface ParsedQuery {
  /** OR of AND-groups. Empty → match-all. */
  alternatives: string[][]
  /** Raw normalized query (trim+lower). */
  raw: string
}

/**
 * Parse user query into OR-of-AND groups.
 * - `foo bar` → [[foo, bar]] (AND)
 * - `foo OR bar` / `foo | bar` → [[foo], [bar]]
 * - `"exact phrase"` kept as one term
 * - trailing `*` retained on the term (prefix-only signal)
 */
export function parseQuery(query: string): ParsedQuery {
  const raw = normalize(query)

  if (!raw) {
    return { alternatives: [], raw: '' }
  }

  // Split on OR / | outside quotes
  const groups: string[] = []
  let buf = ''
  let inQuote = false

  for (let i = 0; i < raw.length; i++) {
    const ch = raw[i]

    if (ch === '"') {
      inQuote = !inQuote
      buf += ch
      continue
    }

    if (!inQuote) {
      if (ch === '|') {
        if (buf.trim()) {
          groups.push(buf.trim())
        }

        buf = ''
        continue
      }

      // word-boundary "or"
      if (
        (ch === 'o' || ch === 'O') &&
        raw.slice(i, i + 2).toLowerCase() === 'or' &&
        isBoundary(raw, i - 1) &&
        isBoundary(raw, i + 2)
      ) {
        if (buf.trim()) {
          groups.push(buf.trim())
        }

        buf = ''
        i += 1
        continue
      }
    }

    buf += ch
  }

  if (buf.trim()) {
    groups.push(buf.trim())
  }

  const alternatives = groups.map(group => tokenizeGroup(group)).filter(terms => terms.length > 0)

  return { alternatives: alternatives.length ? alternatives : [], raw }
}

function isBoundary(s: string, idx: number): boolean {
  if (idx < 0 || idx >= s.length) {
    return true
  }

  return /\s/.test(s[idx])
}

function tokenizeGroup(group: string): string[] {
  const terms: string[] = []
  const re = /"([^"]+)"|(\S+)/g
  let m: RegExpExecArray | null

  while ((m = re.exec(group))) {
    const term = (m[1] ?? m[2] ?? '').trim()

    if (term && term.toLowerCase() !== 'or' && term !== '|') {
      terms.push(term)
    }
  }

  return terms
}

// ── Core scoring ───────────────────────────────────────────────────────────

interface TermHit {
  kind: MatchKind
  score: number
  ranges: Array<[number, number]>
}

function wordsOf(text: string): Array<{ word: string; start: number }> {
  const out: Array<{ word: string; start: number }> = []
  const re = /[\p{L}\p{N}]+/gu
  let m: RegExpExecArray | null

  while ((m = re.exec(text))) {
    out.push({ word: m[0], start: m.index })
  }

  return out
}

function scoreTermInText(haystackLower: string, term: string, allowFuzzy: boolean): TermHit | null {
  if (!term) {
    return null
  }

  const prefixOnly = term.endsWith('*')
  const needle = prefixOnly ? term.slice(0, -1) : term

  if (!needle) {
    return null
  }

  if (haystackLower === needle) {
    return { kind: 'exact', score: GRADE.exact, ranges: [[0, haystackLower.length]] }
  }

  if (haystackLower.startsWith(needle)) {
    return { kind: 'prefix', score: GRADE.prefix, ranges: [[0, needle.length]] }
  }

  const words = wordsOf(haystackLower)

  for (const { word, start } of words) {
    if (word === needle) {
      return { kind: 'word', score: GRADE.word, ranges: [[start, start + word.length]] }
    }
  }

  for (const { word, start } of words) {
    if (word.startsWith(needle)) {
      return { kind: 'word-prefix', score: GRADE['word-prefix'], ranges: [[start, start + needle.length]] }
    }
  }

  if (!prefixOnly) {
    const idx = haystackLower.indexOf(needle)

    if (idx >= 0) {
      return { kind: 'substring', score: GRADE.substring, ranges: [[idx, idx + needle.length]] }
    }
  }

  if (!allowFuzzy || prefixOnly) {
    return null
  }

  // Light fuzzy: only against individual words, bounded edit distance.
  const maxDist = needle.length >= 6 ? 2 : needle.length >= 4 ? 1 : 0

  if (maxDist <= 0) {
    return null
  }

  let best: TermHit | null = null

  for (const { word, start } of words) {
    if (Math.abs(word.length - needle.length) > maxDist) {
      continue
    }

    const d = levenshtein(word, needle, maxDist)

    if (d === null || d <= 0) {
      continue
    }

    const score = GRADE.fuzzy * (1 - d / (maxDist + 1))

    if (!best || score > best.score) {
      best = { kind: 'fuzzy', score, ranges: [[start, start + word.length]] }
    }
  }

  return best
}

/** Bounded Levenshtein; returns null if distance would exceed maxDist. */
export function levenshtein(a: string, b: string, maxDist: number): number | null {
  if (a === b) {
    return 0
  }

  const n = a.length
  const m = b.length

  if (Math.abs(n - m) > maxDist) {
    return null
  }

  if (n === 0) {
    return m <= maxDist ? m : null
  }

  if (m === 0) {
    return n <= maxDist ? n : null
  }

  let prev = new Array<number>(m + 1)
  let curr = new Array<number>(m + 1)

  for (let j = 0; j <= m; j++) {
    prev[j] = j
  }

  for (let i = 1; i <= n; i++) {
    curr[0] = i
    let rowMin = curr[0]

    for (let j = 1; j <= m; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1
      curr[j] = Math.min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
      rowMin = Math.min(rowMin, curr[j])
    }

    if (rowMin > maxDist) {
      return null
    }

    ;[prev, curr] = [curr, prev]
  }

  const dist = prev[m]

  return dist <= maxDist ? dist : null
}

function fieldWeight(field: SearchField, explicit?: number): number {
  return explicit ?? DEFAULT_FIELD_WEIGHT[field] ?? 1
}

function scoreFieldsAgainstTerms(
  fields: FieldSpec[],
  terms: string[],
  allowFuzzy: boolean
): { score: number; matches: FieldMatch[] } | null {
  if (!terms.length) {
    return { score: 1, matches: [] }
  }

  const prepared = fields
    .map(f => ({
      field: f.field,
      weight: fieldWeight(f.field, f.weight),
      raw: asText(f.value),
      lower: asText(f.value).toLowerCase()
    }))
    .filter(f => f.lower.length > 0)

  if (!prepared.length) {
    return null
  }

  const matches: FieldMatch[] = []
  let total = 0

  for (const term of terms) {
    let best: FieldMatch | null = null

    for (const f of prepared) {
      const hit = scoreTermInText(f.lower, term, allowFuzzy && f.field !== 'id')

      if (!hit) {
        continue
      }

      // Keywords field never outranks a visible-label hit of the same grade.
      const kind = f.field === 'keywords' && hit.kind !== 'exact' ? 'keyword' : hit.kind
      const score = (kind === 'keyword' ? GRADE.keyword : hit.score) * f.weight

      if (!best || score > best.score) {
        best = {
          field: f.field,
          kind,
          score,
          value: f.raw,
          ranges: hit.ranges
        }
      }
    }

    if (!best) {
      return null
    }

    matches.push(best)
    total += best.score
  }

  // Prefer fewer, stronger field hits; average keeps multi-term fair.
  const score = total / terms.length

  // Deduplicate matches by field keeping best + merge ranges.
  const byField = new Map<SearchField, FieldMatch>()

  for (const m of matches) {
    const prev = byField.get(m.field)

    if (!prev || m.score > prev.score) {
      const ranges = prev ? mergeRanges([...prev.ranges, ...m.ranges]) : m.ranges
      byField.set(m.field, { ...m, ranges })
    } else if (prev) {
      byField.set(m.field, { ...prev, ranges: mergeRanges([...prev.ranges, ...m.ranges]) })
    }
  }

  return { score, matches: [...byField.values()].sort((a, b) => b.score - a.score) }
}

function mergeRanges(ranges: Array<[number, number]>): Array<[number, number]> {
  if (ranges.length <= 1) {
    return ranges
  }

  const sorted = [...ranges].sort((a, b) => a[0] - b[0] || a[1] - b[1])
  const out: Array<[number, number]> = [[sorted[0][0], sorted[0][1]]]

  for (let i = 1; i < sorted.length; i++) {
    const last = out[out.length - 1]
    const cur = sorted[i]

    if (cur[0] <= last[1]) {
      last[1] = Math.max(last[1], cur[1])
    } else {
      out.push([cur[0], cur[1]])
    }
  }

  return out
}

function fuzzyEnabled(opts?: RankOptions): boolean {
  if (opts?.fuzzy === false) {
    return false
  }

  if (opts?.fuzzy === true) {
    return true
  }

  const max = opts?.fuzzyMaxItems ?? 2000
  const count = opts?.itemCount

  return count == null || count <= max
}

/**
 * Rank one item's fields against a query. Returns null when no alternative matches.
 */
export function rankFields(
  fields: FieldSpec[],
  query: string,
  opts?: RankOptions
): { score: number; matches: FieldMatch[] } | null {
  const parsed = parseQuery(query)

  if (!parsed.alternatives.length) {
    return { score: 1, matches: [] }
  }

  const allowFuzzy = fuzzyEnabled(opts)
  let best: { score: number; matches: FieldMatch[] } | null = null

  for (const terms of parsed.alternatives) {
    const hit = scoreFieldsAgainstTerms(fields, terms, allowFuzzy)

    if (hit && (!best || hit.score > best.score)) {
      best = hit
    }
  }

  return best
}

/** Filter + sort a list by multi-field rank. Stable for equal scores. */
export function rankItems<T>(
  items: readonly T[],
  getFields: (item: T) => FieldSpec[],
  query: string,
  opts?: RankOptions
): Array<RankedHit<T>> {
  const needle = normalize(query)

  if (!needle) {
    return items.map(item => ({ item, score: 1, matches: [] }))
  }

  const baseOpts: RankOptions = { ...opts, itemCount: opts?.itemCount ?? items.length }
  const hits: Array<RankedHit<T> & { idx: number }> = []

  for (let i = 0; i < items.length; i++) {
    const item = items[i]
    const ranked = rankFields(getFields(item), needle, baseOpts)

    if (ranked && ranked.score > 0) {
      hits.push({ item, score: ranked.score, matches: ranked.matches, idx: i })
    }
  }

  hits.sort((a, b) => b.score - a.score || a.idx - b.idx)

  return hits.map(({ item, score, matches }) => ({ item, score, matches }))
}

/** Palette-compatible scorer: label + keywords, returns numeric score (0 = no match). */
export function scoreLabeledItem(
  label: string,
  keywords: string[] | undefined,
  query: string,
  opts?: RankOptions
): number {
  const ranked = rankFields(
    [
      { field: 'label', value: label, weight: 1 },
      { field: 'keywords', value: (keywords ?? []).join(' '), weight: 0.4 }
    ],
    query,
    opts
  )

  return ranked?.score ?? 0
}

export function bestMatch(matches: FieldMatch[]): FieldMatch | undefined {
  return matches[0]
}

// ── Highlight helpers ──────────────────────────────────────────────────────

export function wrapRanges(
  text: string,
  ranges: Array<[number, number]>,
  open = HL_OPEN,
  close = HL_CLOSE
): string {
  if (!ranges.length || !text) {
    return text
  }

  const merged = mergeRanges(ranges)
  let out = ''
  let cursor = 0

  for (const [start, end] of merged) {
    const s = Math.max(0, Math.min(text.length, start))
    const e = Math.max(s, Math.min(text.length, end))

    out += text.slice(cursor, s)
    out += open + text.slice(s, e) + close
    cursor = e
  }

  out += text.slice(cursor)

  return out
}

/** Strip FTS or client highlight markers. */
export function stripHighlightMarkers(text: string): string {
  return text
    .replaceAll(FTS_MARK_OPEN, '')
    .replaceAll(FTS_MARK_CLOSE, '')
    .replaceAll(HL_OPEN, '')
    .replaceAll(HL_CLOSE, '')
}

export interface HighlightSegment {
  text: string
  hit: boolean
}

/** Parse text with either FTS `>>>/<<<` or client `[[m]]` markers into segments. */
export function parseHighlightSegments(text: string): HighlightSegment[] {
  if (!text) {
    return []
  }

  const hasFts = text.includes(FTS_MARK_OPEN)
  const hasHl = text.includes(HL_OPEN)

  if (!hasFts && !hasHl) {
    return [{ text, hit: false }]
  }

  const open = hasFts ? FTS_MARK_OPEN : HL_OPEN
  const close = hasFts ? FTS_MARK_CLOSE : HL_CLOSE
  const segments: HighlightSegment[] = []
  let rest = text

  while (rest.length) {
    const i = rest.indexOf(open)

    if (i < 0) {
      segments.push({ text: rest, hit: false })
      break
    }

    if (i > 0) {
      segments.push({ text: rest.slice(0, i), hit: false })
    }

    rest = rest.slice(i + open.length)
    const j = rest.indexOf(close)

    if (j < 0) {
      segments.push({ text: rest, hit: true })
      break
    }

    segments.push({ text: rest.slice(0, j), hit: true })
    rest = rest.slice(j + close.length)
  }

  return segments.filter(s => s.text.length > 0)
}

/** Build a short snippet around the first highlight / range. */
export function excerptAround(
  text: string,
  ranges: Array<[number, number]>,
  radius = 28
): { text: string; ranges: Array<[number, number]> } {
  if (!text) {
    return { text: '', ranges: [] }
  }

  if (!ranges.length) {
    const clipped = text.length > radius * 2 ? `${text.slice(0, radius * 2)}…` : text

    return { text: clipped, ranges: [] }
  }

  const [start, end] = ranges[0]
  const from = Math.max(0, start - radius)
  const to = Math.min(text.length, end + radius)
  let snippet = text.slice(from, to)
  const adj: Array<[number, number]> = ranges
    .map(([s, e]) => [s - from, e - from] as [number, number])
    .filter(([s, e]) => e > 0 && s < snippet.length)
    .map(([s, e]) => [Math.max(0, s), Math.min(snippet.length, e)] as [number, number])

  if (from > 0) {
    snippet = `…${snippet}`

    for (const r of adj) {
      r[0] += 1
      r[1] += 1
    }
  }

  if (to < text.length) {
    snippet = `${snippet}…`
  }

  return { text: snippet, ranges: adj }
}

/** Human-readable default field labels (EN). UI may override via i18n. */
export const SEARCH_FIELD_LABEL_EN: Record<SearchField, string> = {
  title: 'Title',
  name: 'Name',
  label: 'Label',
  preview: 'Preview',
  description: 'Description',
  category: 'Category',
  cwd: 'Path',
  branch: 'Branch',
  id: 'ID',
  source: 'Source',
  keywords: 'Keywords',
  body: 'Message',
  tool: 'Tool'
}

export const SEARCH_FIELD_LABEL_ZH: Record<SearchField, string> = {
  title: '标题',
  name: '名称',
  label: '标签',
  preview: '预览',
  description: '描述',
  category: '分类',
  cwd: '路径',
  branch: '分支',
  id: '编号',
  source: '来源',
  keywords: '关键词',
  body: '正文',
  tool: '工具'
}
