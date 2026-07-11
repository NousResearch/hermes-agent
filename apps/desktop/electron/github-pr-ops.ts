import { runGh } from './gh'

export const SUMMARY_FIELDS =
  'number,title,url,state,isDraft,updatedAt,createdAt,repository,author,labels,commentsCount'
export const DETAIL_FIELDS = `${SUMMARY_FIELDS},body,headRefName,baseRefName,additions,deletions,changedFiles,reviewDecision,mergeStateStatus,mergedAt,statusCheckRollup`
const REPOSITORY_RE = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/

export function validatePullRequestFilter(filter) {
  if (!filter || !['created', 'review-requested'].includes(filter.kind)) {
    throw new Error('Unsupported pull request filter')
  }

  if (!['open', 'closed'].includes(filter.state)) {
    throw new Error('Unsupported pull request state')
  }

  if (filter.kind === 'review-requested' && filter.state !== 'open') {
    throw new Error('Review-requested only supports open')
  }

  return {
    ...filter,
    limit: Math.min(100, Math.max(1, Number.isFinite(filter.limit) ? Math.trunc(filter.limit) : 100))
  }
}

export function validatePullRequestRef(ref) {
  if (!ref || !REPOSITORY_RE.test(ref.repository || '')) {
    throw new Error('Invalid GitHub repository')
  }

  if (!Number.isSafeInteger(ref.number) || ref.number <= 0) {
    throw new Error('Invalid pull request number')
  }

  return ref
}

export function buildPullRequestSearchArgs(filter) {
  const safe = validatePullRequestFilter(filter)
  const actor = safe.kind === 'created' ? ['--author', '@me'] : ['--review-requested', '@me']

  return [
    'search',
    'prs',
    ...actor,
    '--state',
    safe.state,
    '--sort',
    'updated',
    '--order',
    'desc',
    '--limit',
    String(safe.limit),
    '--json',
    SUMMARY_FIELDS
  ]
}

function repositoryName(raw) {
  const value = typeof raw === 'string' ? raw : raw?.nameWithOwner

  return REPOSITORY_RE.test(value || '') ? value : null
}

function normalizeLabels(labels) {
  return Array.isArray(labels)
    ? labels
        .filter(x => x && typeof x.name === 'string')
        .map(x => ({ name: x.name, ...(typeof x.color === 'string' ? { color: x.color } : {}) }))
    : []
}

function normalizeAuthor(author) {
  return author && typeof author.login === 'string'
    ? { login: author.login, ...(typeof author.url === 'string' ? { url: author.url } : {}) }
    : null
}

export function normalizePullRequestSummary(raw) {
  const repository = repositoryName(raw?.repository)
  const number = Number(raw?.number)

  if (
    !repository ||
    !Number.isSafeInteger(number) ||
    number <= 0 ||
    typeof raw?.title !== 'string' ||
    typeof raw?.url !== 'string'
  ) {
    return null
  }
  const state = ['OPEN', 'CLOSED', 'MERGED'].includes(String(raw.state).toUpperCase())
    ? String(raw.state).toUpperCase()
    : 'UNKNOWN'

  return {
    id: `${repository}#${number}`,
    repository,
    number,
    title: raw.title,
    url: raw.url,
    state,
    isDraft: Boolean(raw.isDraft),
    author: normalizeAuthor(raw.author),
    labels: normalizeLabels(raw.labels),
    commentsCount: Math.max(0, Number(raw.commentsCount) || 0),
    createdAt: typeof raw.createdAt === 'string' ? raw.createdAt : '',
    updatedAt: typeof raw.updatedAt === 'string' ? raw.updatedAt : ''
  }
}

export function normalizeCheckSummary(checks) {
  const result = { total: 0, pending: 0, passed: 0, failed: 0, skipped: 0 }

  for (const check of Array.isArray(checks) ? checks : []) {
    result.total++
    const status = String(check?.status || '').toUpperCase()
    const conclusion = String(check?.conclusion || check?.state || '').toUpperCase()

    if (status && status !== 'COMPLETED') {
      result.pending++
    } else if (['SUCCESS', 'NEUTRAL'].includes(conclusion)) {
      result.passed++
    } else if (['SKIPPED', 'STALE'].includes(conclusion)) {
      result.skipped++
    } else if (['FAILURE', 'ERROR', 'CANCELLED', 'TIMED_OUT', 'ACTION_REQUIRED'].includes(conclusion)) {
      result.failed++
    } else {
      result.pending++
    }
  }

  return result
}

export function normalizePullRequestDetail(raw, repository) {
  const summary = normalizePullRequestSummary({ ...raw, repository })

  if (!summary) {
    return null
  }
  const mergedAt = typeof raw.mergedAt === 'string' ? raw.mergedAt : null

  return {
    ...summary,
    state: mergedAt ? 'MERGED' : summary.state,
    body: typeof raw.body === 'string' ? raw.body : '',
    headRefName: typeof raw.headRefName === 'string' ? raw.headRefName : '',
    baseRefName: typeof raw.baseRefName === 'string' ? raw.baseRefName : '',
    additions: Math.max(0, Number(raw.additions) || 0),
    deletions: Math.max(0, Number(raw.deletions) || 0),
    changedFiles: Math.max(0, Number(raw.changedFiles) || 0),
    reviewDecision: typeof raw.reviewDecision === 'string' && raw.reviewDecision ? raw.reviewDecision : null,
    mergeStateStatus: typeof raw.mergeStateStatus === 'string' && raw.mergeStateStatus ? raw.mergeStateStatus : null,
    mergedAt,
    checks: normalizeCheckSummary(raw.statusCheckRollup)
  }
}

export async function getGithubAuthStatus(ghBin, runner = runGh) {
  const result = await runner(['auth', 'status'], { ghBin })

  if (result.kind === 'missing') {
    return 'gh-missing'
  }

  return result.ok ? 'ready' : 'not-authenticated'
}

export async function listGithubPullRequests(filter, ghBin, runner = runGh) {
  const authState = await getGithubAuthStatus(ghBin, runner)

  if (authState !== 'ready') {
    return { authState, items: [], fetchedAt: Date.now() }
  }
  const result = await runner(buildPullRequestSearchArgs(filter), { ghBin })

  if (!result.ok) {
    return {
      authState: 'error',
      items: [],
      fetchedAt: Date.now(),
      error: result.kind === 'timeout' ? 'GitHub CLI request timed out' : 'Failed to load pull requests'
    }
  }

  try {
    const raw = JSON.parse(result.stdout || '')

    if (!Array.isArray(raw)) {
      throw new Error('Expected an array')
    }

    return { authState: 'ready', items: raw.map(normalizePullRequestSummary).filter(Boolean), fetchedAt: Date.now() }
  } catch {
    return { authState: 'error', items: [], fetchedAt: Date.now(), error: 'GitHub CLI returned invalid data' }
  }
}

export async function getGithubPullRequestDetail(ref, ghBin, runner = runGh) {
  const safe = validatePullRequestRef(ref)
  const result = await runner(['pr', 'view', String(safe.number), '--repo', safe.repository, '--json', DETAIL_FIELDS], {
    ghBin
  })

  if (!result.ok) {
    throw new Error(result.kind === 'timeout' ? 'GitHub CLI request timed out' : 'Failed to load pull request details')
  }
  let raw

  try {
    raw = JSON.parse(result.stdout || '')
  } catch {
    throw new Error('GitHub CLI returned invalid data')
  }
  const detail = normalizePullRequestDetail(raw, safe.repository)

  if (!detail) {
    throw new Error('GitHub CLI returned invalid pull request data')
  }

  return detail
}
