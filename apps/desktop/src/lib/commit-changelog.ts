/**
 * Tiny user-facing changelog builder. Takes a list of raw commit summaries,
 * parses the Conventional Commits 1.0 header (`type(scope)!: subject`),
 * filters internal noise (chore/ci/docs/...), and groups the rest into
 * friendly buckets for end users (What's new, Fixed, Faster, Improved).
 *
 * Inlined (rather than depending on `conventional-commits-parser`) because
 * that package's index re-exports a Node `stream` helper which won't load
 * in the sandboxed Electron renderer, and its actual parse logic for the
 * header is a small regex.
 */

export type CommitGroupId = 'new' | 'fixed' | 'faster' | 'improved' | 'other'

export interface CommitGroup {
  id: CommitGroupId
  label: string
  items: string[]
}

export interface ParsedCommit {
  type: null | string
  scope: null | string
  breaking: boolean
  subject: string
}

export interface CommitChangelogInput {
  summary?: string
}

interface BuildOptions {
  maxGroups?: number
  maxPerGroup?: number
  maxTotal?: number
}

const GROUP_META: Record<CommitGroupId, { label: string; order: number }> = {
  new: { label: "What's new", order: 0 },
  fixed: { label: 'Fixed', order: 1 },
  faster: { label: 'Faster', order: 2 },
  improved: { label: 'Improved', order: 3 },
  other: { label: 'Other improvements', order: 4 }
}

const TYPE_TO_GROUP: Record<string, CommitGroupId> = {
  feat: 'new',
  feature: 'new',
  fix: 'fixed',
  bugfix: 'fixed',
  hotfix: 'fixed',
  revert: 'fixed',
  perf: 'faster',
  performance: 'faster',
  refactor: 'improved',
  a11y: 'improved',
  ui: 'improved',
  ux: 'improved'
}

const HIDDEN_TYPES = new Set([
  'build',
  'chore',
  'ci',
  'dep',
  'deps',
  'doc',
  'docs',
  'lint',
  'release',
  'style',
  'test',
  'tests',
  'wip'
])

const FALLBACK_GROUP: CommitGroup = { id: 'other', items: ['Improvements and fixes'], label: 'In this update' }

const CONVENTIONAL_HEADER = /^(?<type>[a-zA-Z][a-zA-Z0-9_-]*)(?:\((?<scope>[^)]+)\))?(?<bang>!)?:\s+(?<subject>.+)$/

/** Parse a single commit header line per Conventional Commits 1.0. */
export function parseCommitHeader(raw: string): ParsedCommit {
  const header = (raw ?? '').split(/\r?\n/, 1)[0].trim()

  if (!header) {
    return { breaking: false, scope: null, subject: '', type: null }
  }

  const match = CONVENTIONAL_HEADER.exec(header)

  if (!match?.groups) {
    return { breaking: false, scope: null, subject: header, type: null }
  }

  return {
    breaking: Boolean(match.groups.bang),
    scope: match.groups.scope ?? null,
    subject: match.groups.subject.trim(),
    type: match.groups.type.toLowerCase()
  }
}

function tidySubject(subject: string): string {
  const cleaned = subject
    .replace(/\s+/g, ' ')
    .replace(/[.;,\s]+$/, '')
    .trim()

  if (!cleaned) {
    return cleaned
  }

  return cleaned.charAt(0).toUpperCase() + cleaned.slice(1)
}

/**
 * Build a small grouped changelog from a list of raw commits.
 * Always returns at least one group; falls back to a neutral placeholder
 * when every commit was filtered or unparseable.
 */
export function buildCommitChangelog(
  commits: readonly CommitChangelogInput[] | undefined,
  options: BuildOptions = {}
): CommitGroup[] {
  const { maxGroups = 3, maxPerGroup = 4, maxTotal = 6 } = options
  const groups = new Map<CommitGroupId, string[]>()
  const seen = new Set<string>()
  let total = 0

  for (const commit of commits ?? []) {
    if (total >= maxTotal) {
      break
    }

    const parsed = parseCommitHeader(commit.summary ?? '')

    if (parsed.type && HIDDEN_TYPES.has(parsed.type)) {
      continue
    }

    const groupId: CommitGroupId = parsed.type ? (TYPE_TO_GROUP[parsed.type] ?? 'other') : 'other'
    const subject = tidySubject(parsed.subject)

    if (!subject) {
      continue
    }

    const dedupeKey = subject.toLowerCase()

    if (seen.has(dedupeKey)) {
      continue
    }

    const bucket = groups.get(groupId) ?? []

    if (bucket.length >= maxPerGroup) {
      continue
    }

    bucket.push(subject)
    groups.set(groupId, bucket)
    seen.add(dedupeKey)
    total += 1
  }

  const result = Array.from(groups.entries())
    .map(([id, items]) => ({ id, items, label: GROUP_META[id].label, order: GROUP_META[id].order }))
    .sort((a, b) => a.order - b.order)
    .slice(0, maxGroups)
    .map(({ id, items, label }): CommitGroup => ({ id, items, label }))

  if (result.length === 0) {
    return [FALLBACK_GROUP]
  }

  return result
}
