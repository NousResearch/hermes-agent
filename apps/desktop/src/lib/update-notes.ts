/**
 * "What's in this update" for the updates overlay, read from public GitHub.
 *
 * Stable channel: the release's own notes (Highlights headlines, else its first
 * bullet list). Main: the newest merged PR titles. Both are best-effort; the
 * caller keeps the commit changelog when nothing comes back.
 */

const REPOSITORY = 'NousResearch/hermes-agent'
const API = `https://api.github.com/repos/${REPOSITORY}`
const SITE = `https://github.com/${REPOSITORY}`
const MAX_ITEMS = 5
const MAX_CHARS = 140

export type UpdateNotesSource = { kind: 'release'; tag: null | string } | { kind: 'prs' }

interface NotesStatus {
  branch?: string
  channel?: string
  commits?: readonly { sha?: string }[]
  behind?: null | number
  currentSha?: string
  latestTag?: null | string
  rollback?: boolean
  sourceVersion?: string
  targetSha?: string
}

interface PullRow {
  title?: string
  merged_at?: null | string
  merge_commit_sha?: null | string
}

/** The release a stable update installs: packaged feeds name a tag, source checks a version. */
export function updateReleaseTag(status: NotesStatus): null | string {
  return status.latestTag || (status.channel === 'stable' && status.sourceVersion ? `v${status.sourceVersion}` : null)
}

/** Which notes describe this update; null when neither fits (custom branch, canary, rollback). */
export function updateNotesSource(status: NotesStatus): null | UpdateNotesSource {
  if (status.rollback) {
    return null
  }

  if (status.channel === 'stable') {
    return { kind: 'release', tag: updateReleaseTag(status) }
  }

  return status.branch === 'main' || status.channel === 'main' ? { kind: 'prs' } : null
}

const isSha = (value?: string): value is string => Boolean(value && /^[0-9a-f]{7,40}$/i.test(value))

/** Where "more info" lives: the release page, or the compare view of exactly this update. */
export function updateNotesUrl(source: UpdateNotesSource, status: NotesStatus): string {
  if (source.kind === 'release') {
    return source.tag ? `${SITE}/releases/tag/${encodeURIComponent(source.tag)}` : `${SITE}/releases/latest`
  }

  return isSha(status.currentSha) && isSha(status.targetSha)
    ? `${SITE}/compare/${status.currentSha}...${status.targetSha}`
    : `${SITE}/commits/main`
}

function clip(text: string): string {
  const flat = text.replace(/\s+/g, ' ').trim()

  return flat.length <= MAX_CHARS ? flat : `${flat.slice(0, MAX_CHARS - 1).replace(/\s+\S*$/, '')}…`
}

function plain(markdown: string): string {
  return markdown
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\*\*|__|`/g, '')
    .replace(/\s*\([^()]*#\d+[^()]*\)\s*$/, '')
    .replace(/[\s.:;,]+$/, '')
}

// How-to-install and credits sections are not "what changed".
const SKIPPED_SECTION = /updating|upgrad|install|contributor|changelog/i

/** Up to five headlines: the Highlights section's bullets, else the body's first change list. */
export function extractReleaseHighlights(body: string, max = MAX_ITEMS): string[] {
  const lines = body.split(/\r?\n/)
  const start = lines.findIndex(line => /^#{1,6}\s.*highlights/i.test(line))
  const items: string[] = []
  let skipped = false

  for (const line of start >= 0 ? lines.slice(start + 1) : lines) {
    const heading = /^#{1,6}\s+(.*)$/.exec(line)

    if (heading) {
      if (start >= 0 || items.length > 0) {
        break
      }

      skipped = SKIPPED_SECTION.test(heading[1])

      continue
    }

    const bullet = skipped ? null : /^[-*]\s+(.+)$/.exec(line)

    if (!bullet) {
      continue
    }

    // A bold lead-in is the item's own headline; the prose after it is detail.
    const text = clip(plain(/^\*\*(.+?)\*\*/.exec(bullet[1])?.[1] ?? bullet[1]))

    if (text && items.push(text) >= max) {
      break
    }
  }

  return items
}

/**
 * The newest merged PRs. A complete commit list pins them to this update exactly
 * (a merged PR's merge_commit_sha is a commit on main); otherwise the newest
 * merges stand in for it.
 */
export function latestMergedPrTitles(
  pulls: readonly PullRow[],
  commits: readonly { sha?: string }[],
  behind: null | number | undefined,
  max = MAX_ITEMS
): string[] {
  const shas = commits.map(commit => commit.sha).filter(isSha)
  const complete = shas.length > 0 && behind === shas.length

  return pulls
    .filter(pull => pull.merged_at && pull.title?.trim())
    .filter(pull => !complete || shas.some(sha => pull.merge_commit_sha?.startsWith(sha)))
    .sort((a, b) => String(b.merged_at).localeCompare(String(a.merged_at)))
    .slice(0, max)
    .map(pull => clip(pull.title ?? ''))
}

async function getJson(url: string): Promise<unknown> {
  const response = await fetch(url, {
    headers: { Accept: 'application/vnd.github+json' },
    signal: AbortSignal.timeout(8000)
  })

  if (!response.ok) {
    throw new Error(`GitHub answered ${response.status} for ${url}`)
  }

  return response.json()
}

const cache = new Map<string, Promise<string[]>>()

/** The notes' items; [] when GitHub is unreachable or has none. Cached per target. */
export function fetchUpdateNoteItems(source: UpdateNotesSource, status: NotesStatus): Promise<string[]> {
  const key =
    source.kind === 'release' ? `release:${source.tag ?? 'latest'}` : `prs:${status.currentSha}:${status.targetSha}`

  const cached = cache.get(key)

  if (cached) {
    return cached
  }

  const pending = (
    source.kind === 'release'
      ? getJson(source.tag ? `${API}/releases/tags/${encodeURIComponent(source.tag)}` : `${API}/releases/latest`).then(
          release => extractReleaseHighlights(String((release as { body?: unknown })?.body ?? ''))
        )
      : // Recently updated closed PRs hold the newest merges; one request, no auth.
        getJson(`${API}/pulls?state=closed&base=main&sort=updated&direction=desc&per_page=100`).then(pulls =>
          latestMergedPrTitles(Array.isArray(pulls) ? pulls : [], status.commits ?? [], status.behind)
        )
  ).catch(() => {
    cache.delete(key)

    return []
  })

  cache.set(key, pending)

  return pending
}
