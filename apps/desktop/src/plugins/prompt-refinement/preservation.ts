/**
 * What a rewrite must keep. Deterministic checks over the original draft:
 * each protected item found there must still be present in the candidate
 * (the refined text, or the user's edit of it) before Approve is allowed.
 *
 * Two severities:
 *   blocking — exact content (code, @references, URLs, paths, model/agent
 *              identifiers, numbers). Approve stays disabled until the text
 *              contains it again; the user fixes it with Edit or sends the
 *              original instead.
 *   review   — heuristics that can misfire on a legitimate rephrase (fewer
 *              negations or requirement words, hyphenated names). Approve
 *              needs an explicit "I've checked this" acknowledgement.
 *
 * The checker under-approximates on purpose: it never claims meaning is
 * preserved, it only guarantees these items were not silently dropped.
 * Intent checks (intent.ts) add review-only findings for what the rewrite asks
 * for that the draft did not: destructive steps, installs, outside contact,
 * spending, permissions, delegation changes, text carried in from the chat.
 */

import { checkIntent, type IntentKind } from './intent'

export type PreservedKind =
  | IntentKind
  | 'code-block'
  | 'identifier'
  | 'inline-code'
  | 'name'
  | 'negation'
  | 'number'
  | 'path'
  | 'reference'
  | 'requirement'
  | 'url'

export type IssueSeverity = 'blocking' | 'review'

export interface PreservationIssue {
  kind: PreservedKind
  severity: IssueSeverity
  /** The missing item, or for word classes the count that went missing. */
  value: string
}

export interface PreservationReport {
  issues: PreservationIssue[]
  blocking: number
  review: number
}

const FENCE_RE = /(```|~~~)[^\n]*\n([\s\S]*?)\1/g
const REFERENCE_RE = /(?<![\w/])@[a-z][a-z0-9-]*:(?:`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|\S+)/gi
const INLINE_CODE_RE = /`([^`\n]+)`/g
const URL_RE = /\b[a-z][a-z0-9+.-]*:\/\/[^\s<>"'`]+/gi
const TRAILING_PUNCT_RE = /[.,;:!?)\]}'"»”’]+$/
const LEADING_PUNCT_RE = /^[([{'"«“‘]+/

const FILE_EXTENSIONS = new Set(
  (
    'ts tsx js jsx mjs cjs json jsonc md mdx py pyi rs go java kt kts swift c h cc cpp hpp cs rb php sh zsh bash fish ' +
    'ps1 yml yaml toml ini cfg conf env lock txt csv tsv html htm css scss less sql xml svg png jpg jpeg gif webp pdf ' +
    'doc docx xls xlsx ppt pptx ipynb vue svelte dart lua r pl ex exs zig nix gradle plist log dockerfile proto graphql'
  ).split(' ')
)

const NEGATION_RE = /\b(?:not|no|never|none|nor|without|avoid|cannot)\b|n['’]t\b/gi
const REQUIREMENT_RE = /\b(?:must|only|exactly|always|required?|ensure|mandatory|strictly)\b/gi

/** Replace a match with spaces of the same length, so later passes skip it
 *  without shifting offsets or gluing neighbouring words together. */
const blank = (text: string, re: RegExp) => text.replace(re, match => ' '.repeat(match.length))

const stripPunct = (token: string) => token.replace(LEADING_PUNCT_RE, '').replace(TRAILING_PUNCT_RE, '')

function isPath(token: string): boolean {
  if (/^[A-Za-z]:\\\S+/.test(token) || /^(?:~|\.{1,2})?\/[^\s/]/.test(token)) {
    return true
  }

  const slashes = (token.match(/\//g) ?? []).length
  const extension = /\.([A-Za-z0-9]{1,10})$/.exec(token)?.[1]?.toLowerCase()
  const hasKnownExtension = Boolean(extension && FILE_EXTENSIONS.has(extension))

  if (slashes >= 1 && /^[\w.@~-]+(?:\/[\w.@~-]+)+\/?$/.test(token)) {
    return slashes >= 2 || hasKnownExtension
  }

  return hasKnownExtension && /^[\w-]+(?:\.[\w-]+)*\.[A-Za-z0-9]{1,10}$/.test(token)
}

/** Model / agent / tool identifiers: versioned slugs (gpt-5.5, claude-opus-5-5,
 *  python3.12, x86_64), letter-led alphanumerics (o3, gpt4o), snake_case,
 *  camelCase and multi-hump PascalCase (PromptRouter). */
function isIdentifier(token: string): boolean {
  return (
    (/[A-Za-z]/.test(token) &&
      /\d/.test(token) &&
      /[-_.:]/.test(token) &&
      /^[A-Za-z0-9][\w.:+-]*[A-Za-z0-9]$/.test(token)) ||
    /^[A-Za-z][A-Za-z]*\d[A-Za-z0-9]*$/.test(token) ||
    /^[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+$/.test(token) ||
    /^[a-z]+(?:[A-Z][a-z0-9]*)+$/.test(token) ||
    /^[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+$/.test(token)
  )
}

/** Lowercase hyphenated names (skill / plugin slugs such as code-review-routing)
 *  look exactly like English compounds (step-by-step), so they are review-only. */
const isKebabName = (token: string) => /^[a-z][a-z0-9]*(?:-[a-z0-9]+)+$/.test(token)

interface Protected {
  kind: PreservedKind
  severity: IssueSeverity
  value: string
}

function collect(original: string): { items: Protected[]; prose: string } {
  const items: Protected[] = []

  const add = (kind: PreservedKind, value: string, severity: IssueSeverity = 'blocking') => {
    if (value.trim()) {
      items.push({ kind, severity, value })
    }
  }

  for (const match of original.matchAll(FENCE_RE)) {
    add('code-block', match[2].replace(/\n$/, ''))
  }

  let rest = blank(original, FENCE_RE)

  for (const match of rest.matchAll(REFERENCE_RE)) {
    add('reference', /^@[^:]+:[`"']/.test(match[0]) ? match[0] : stripPunct(match[0]))
  }

  rest = blank(rest, REFERENCE_RE)

  for (const match of rest.matchAll(INLINE_CODE_RE)) {
    add('inline-code', match[1])
  }

  rest = blank(rest, INLINE_CODE_RE)

  for (const match of rest.matchAll(URL_RE)) {
    add('url', match[0].replace(TRAILING_PUNCT_RE, ''))
  }

  rest = blank(rest, URL_RE)

  for (const raw of rest.split(/\s+/)) {
    const token = stripPunct(raw)

    if (!token) {
      continue
    }

    if (isPath(token)) {
      add('path', token)
    } else if (isIdentifier(token)) {
      add('identifier', token)
    } else if (isKebabName(token)) {
      add('name', token, 'review')
    } else {
      // A number keeps the letters attached to it ("2d/3d" -> 2d, 3d; "60fps"; "4K"),
      // so the check asks for the same word, not a bare digit inside it.
      for (const number of token.match(/[A-Za-z]*\d+(?:[.,:]\d+)*[A-Za-z%]*/g) ?? []) {
        add('number', number)
      }
    }
  }

  return { items, prose: rest }
}

const escapeRe = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/** Present as a whole item: not merely a prefix of a longer word or number. */
function contains(candidate: string, item: Protected): boolean {
  if (item.kind === 'code-block' || item.kind === 'reference' || item.kind === 'url' || item.kind === 'inline-code') {
    return candidate.includes(item.value)
  }

  // Units and suffixes on numbers ("2d" vs "2D") are the same word; names are not.
  const flags = item.kind === 'number' ? 'i' : ''

  return new RegExp(`(?<![\\w.])${escapeRe(item.value)}(?![\\w]|\\.\\d)`, flags).test(candidate)
}

const count = (text: string, re: RegExp) => (text.match(re) ?? []).length

/** Prose of a text with code, references, inline code and URLs blanked out:
 *  word-class counts must not be fooled by words inside code. */
function proseOf(text: string): string {
  return [FENCE_RE, REFERENCE_RE, INLINE_CODE_RE, URL_RE].reduce((acc, re) => blank(acc, re), text)
}

/** `context` is the chat excerpt the refiner saw ('' when none): text copied
 *  from it into the rewrite is flagged. */
export function checkPreservation(original: string, candidate: string, context = ''): PreservationReport {
  const { items, prose } = collect(original)
  const issues: PreservationIssue[] = []
  const seen = new Set<string>()

  for (const item of items) {
    const key = `${item.kind}\u0000${item.value}`

    if (seen.has(key) || contains(candidate, item)) {
      continue
    }

    seen.add(key)
    issues.push({ kind: item.kind, severity: item.severity, value: item.value })
  }

  const candidateProse = proseOf(candidate)

  for (const [kind, re] of [
    ['negation', NEGATION_RE],
    ['requirement', REQUIREMENT_RE]
  ] as const) {
    const missing = count(prose, re) - count(candidateProse, re)

    if (missing > 0) {
      issues.push({ kind, severity: 'review', value: String(missing) })
    }
  }

  for (const issue of checkIntent(original, candidate, context)) {
    issues.push({ kind: issue.kind, severity: 'review', value: issue.value })
  }

  const blocking = issues.filter(issue => issue.severity === 'blocking').length

  return { issues, blocking, review: issues.length - blocking }
}

/** The one approval rule, shared by the dialog (button state) and the review
 *  store (which re-checks it, so no UI path can approve around it). */
export function canApprove(report: PreservationReport, acknowledged: boolean): boolean {
  return report.blocking === 0 && (report.review === 0 || acknowledged)
}
