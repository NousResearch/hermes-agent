// Authenticated GitHub link-title resolution: private issue/PR/repo URLs 404
// for the unauthenticated curl + hidden-renderer tiers in main.ts's
// fetchLinkTitle, so those tiers surface GitHub's "Page not found" chrome
// instead of the real title. When a link parses as a github.com resource we
// shell out to `gh api` (already authenticated via the user's `gh auth
// login`) before falling back to the scraping tiers.

import { execFile } from 'node:child_process'
import path from 'node:path'

const GITHUB_HOSTS = new Set(['github.com', 'www.github.com'])
const OWNER_REPO_RE = /^[A-Za-z0-9_.-]+$/
const PULL_SEGMENTS = new Set(['pull', 'pulls'])
const GITHUB_TITLE_TIMEOUT_MS = 5000
const GITHUB_TITLE_MAX_LENGTH = 240

export type GitHubResource =
  | { kind: 'repo'; owner: string; repo: string }
  | { kind: 'issue'; owner: string; repo: string; number: number }
  | { kind: 'pull'; owner: string; repo: string; number: number }

// Recognizes github.com/www.github.com issue, PR, and repo URLs; everything
// else (lists, actions, blob/tree, gist.github.com, non-github hosts) → null
// so callers skip straight to the curl/renderer tiers without spawning gh.
export function parseGitHubResource(url: string): GitHubResource | null {
  let parsed: URL

  try {
    parsed = new URL(url)
  } catch {
    return null
  }

  if (!GITHUB_HOSTS.has(parsed.hostname.toLowerCase())) {
    return null
  }

  const segments = parsed.pathname.split('/').filter(Boolean)

  if (segments.length !== 2 && segments.length !== 4) {
    return null
  }

  const [owner, repo, section, numberSegment] = segments

  if (!OWNER_REPO_RE.test(owner) || !OWNER_REPO_RE.test(repo)) {
    return null
  }

  if (segments.length === 2) {
    return { kind: 'repo', owner, repo }
  }

  if (!/^\d+$/.test(numberSegment)) {
    return null
  }

  const number = Number(numberSegment)

  if (!Number.isInteger(number) || number <= 0) {
    return null
  }

  if (section === 'issues') {
    return { kind: 'issue', owner, repo, number }
  }

  if (PULL_SEGMENTS.has(section)) {
    return { kind: 'pull', owner, repo, number }
  }

  return null
}

export function githubApiPath(resource: GitHubResource): string {
  const { owner, repo } = resource

  if (resource.kind === 'issue') {
    return `repos/${owner}/${repo}/issues/${resource.number}`
  }

  if (resource.kind === 'pull') {
    return `repos/${owner}/${repo}/pulls/${resource.number}`
  }

  return `repos/${owner}/${repo}`
}

export function titleFromGitHubApiJson(resource: GitHubResource, json: unknown): string {
  if (!json || typeof json !== 'object') {
    return ''
  }

  const record = json as Record<string, unknown>
  const field = resource.kind === 'repo' ? record.name : record.title

  return typeof field === 'string' ? field.trim() : ''
}

// GUI-launched Electron apps on macOS inherit only a minimal PATH (no
// /opt/homebrew/bin or /usr/local/bin), so a bare `gh` spawn ENOENTs even
// though it works in the user's terminal. Same fix as git-review-ops.ts's
// ghEnv, kept local so this module has no main.ts/git-review-ops coupling.
function githubCliEnv(ghBin?: string) {
  const extra = [ghBin ? path.dirname(ghBin) : '', '/opt/homebrew/bin', '/usr/local/bin', '/usr/bin'].filter(
    dir => dir && dir !== '.'
  )

  return { ...process.env, PATH: [...extra, process.env.PATH].filter(Boolean).join(path.delimiter) }
}

interface RunExecFileResult {
  ok: boolean
  stdout: string
}

function defaultRunExecFile(bin: string, args: string[], options: Record<string, unknown>): Promise<RunExecFileResult> {
  return new Promise(resolve => {
    execFile(bin, args, options, (err, stdout) => resolve({ ok: !err, stdout: String(stdout || '') }))
  })
}

export interface FetchGitHubTitleOptions {
  ghBin?: string
  runExecFile?: (bin: string, args: string[], options: Record<string, unknown>) => Promise<RunExecFileResult>
}

// Never throws — gh missing, unauthenticated, rate-limited, or timed out all
// resolve to '' so fetchLinkTitle in main.ts falls through to the next tier.
export async function fetchGitHubTitle(url: string, options: FetchGitHubTitleOptions = {}): Promise<string> {
  const resource = parseGitHubResource(url)

  if (!resource) {
    return ''
  }

  const { ghBin, runExecFile = defaultRunExecFile } = options

  const result = await runExecFile(ghBin || 'gh', ['api', githubApiPath(resource)], {
    env: githubCliEnv(ghBin),
    windowsHide: true,
    timeout: GITHUB_TITLE_TIMEOUT_MS,
    maxBuffer: 1024 * 1024
  }).catch(() => ({ ok: false, stdout: '' }))

  if (!result.ok || !result.stdout.trim()) {
    return ''
  }

  try {
    return titleFromGitHubApiJson(resource, JSON.parse(result.stdout)).slice(0, GITHUB_TITLE_MAX_LENGTH)
  } catch {
    return ''
  }
}

// GitHub's HTML <title> tags append chrome like " · Issue #6 · owner/repo ·
// GitHub" or " · GitHub" — strip it so a curl/renderer-scraped title reads
// the same as the gh-api one. Applied in a loop since the 404 page doubles
// the suffix ("Page not found · GitHub · GitHub").
const GITHUB_TITLE_SUFFIX_RES = [
  /\s*·\s*Issue\s*#\d+\s*·\s*[\w.-]+\/[\w.-]+\s*·\s*GitHub\s*$/i,
  /\s*·\s*Pull Request\s*#\d+\s*·\s*[\w.-]+\/[\w.-]+\s*·\s*GitHub\s*$/i,
  /\s*·\s*[\w.-]+\/[\w.-]+\s*·\s*GitHub\s*$/i,
  /\s*·\s*GitHub\s*$/i
]

export function cleanGitHubHtmlTitle(title: string): string {
  let cleaned = title.trim()
  let previous: string

  do {
    previous = cleaned

    for (const suffixRe of GITHUB_TITLE_SUFFIX_RES) {
      cleaned = cleaned.replace(suffixRe, '').trim()
    }
  } while (cleaned && cleaned !== previous)

  return cleaned
}

// Superset of main.ts's TITLE_ERROR_RE (bot walls/captchas) plus GitHub's
// 404 chrome, which slips through those terms unmatched. Exported so both
// the untestable main.ts entry point and its tests share one definition
// instead of drifting apart.
const UNUSABLE_TITLE_RE =
  /\b(?:access denied|attention required|captcha|error|forbidden|just a moment|page not found|request blocked|too many requests)\b|^not found$/i

export function isUnusableLinkTitle(value: string): boolean {
  return !value || UNUSABLE_TITLE_RE.test(value)
}
