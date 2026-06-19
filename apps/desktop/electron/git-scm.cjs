'use strict'

// Git source-control operations for the desktop Source Control panel.
//
// Mirrors git-root.cjs: a pure, testable module over the git binary. Every
// operation runs `git` via execFile with an ARGUMENT ARRAY (never a shell
// string) so file names / commit messages can't inject. The repo root is
// resolved + hardened through resolveRequestedPathForIpc before any spawn, so
// the renderer can't point us outside an allowed path.
//
// Status uses porcelain=v2 + -z (NUL-delimited) so filenames with spaces,
// quotes, newlines, and unicode parse unambiguously.

const { execFile } = require('node:child_process')

const { findGitRoot } = require('./git-root.cjs')
const { resolveRequestedPathForIpc } = require('./hardening.cjs')

const MAX_BUFFER = 16 * 1024 * 1024 // 16MB — large diffs / status on big repos.
const DEFAULT_TIMEOUT_MS = 20_000

// Resolve a renderer-supplied cwd to a hardened absolute git root, or null.
function resolveRepoRoot(cwd) {
  let resolved
  try {
    resolved = resolveRequestedPathForIpc(cwd, { purpose: 'Git source control' })
  } catch {
    return null
  }

  return findGitRoot(resolved)
}

// Run git with an arg array in `root`. Resolves to { code, stdout, stderr }.
// Never rejects on a non-zero exit — callers decide what an error means (e.g.
// `git diff` exits 1 when there are differences).
function runGit(gitBinary, root, args, options = {}) {
  return new Promise(resolve => {
    execFile(
      gitBinary,
      ['-C', root, ...args],
      {
        cwd: root,
        timeout: options.timeout || DEFAULT_TIMEOUT_MS,
        maxBuffer: MAX_BUFFER,
        windowsHide: true,
        encoding: options.encoding || 'utf8',
        // Keep git non-interactive: never pop a credential prompt that would
        // hang the spawn. Push/pull rely on the OS credential helper / agent.
        env: { ...process.env, GIT_TERMINAL_PROMPT: '0', GIT_OPTIONAL_LOCKS: '0' }
      },
      (error, stdout, stderr) => {
        const code = error && typeof error.code === 'number' ? error.code : error ? 1 : 0
        resolve({
          code,
          stdout: stdout == null ? '' : stdout,
          stderr: stderr == null ? '' : String(stderr),
          timedOut: Boolean(error && error.killed)
        })
      }
    )
  })
}

// XY status code (porcelain v2) → a friendly status word the UI renders.
function statusWord(x, y) {
  const code = (x || ' ') + (y || ' ')
  if (code.includes('U') || x === 'U' || y === 'U') return 'conflicted'
  const c = x !== '.' && x !== ' ' ? x : y
  switch (c) {
    case 'M':
      return 'modified'
    case 'A':
      return 'added'
    case 'D':
      return 'deleted'
    case 'R':
      return 'renamed'
    case 'C':
      return 'copied'
    case 'T':
      return 'typechange'
    default:
      return 'modified'
  }
}

// Parse `git status --porcelain=v2 -z --branch` into structured groups.
// Returns { branch, upstream, ahead, behind, staged[], unstaged[], untracked[],
// conflicted[] } where each entry is { path, origPath?, status, staged }.
function parseStatus(z) {
  const out = {
    branch: null,
    upstream: null,
    ahead: 0,
    behind: 0,
    staged: [],
    unstaged: [],
    untracked: [],
    conflicted: []
  }

  const records = z.split('\u0000')
  let i = 0

  while (i < records.length) {
    const rec = records[i]
    if (!rec) {
      i += 1
      continue
    }

    const kind = rec[0]

    if (kind === '#') {
      // Branch header line, e.g. "# branch.head main" / "# branch.ab +1 -2".
      const m = rec.slice(2)
      if (m.startsWith('branch.head ')) out.branch = m.slice('branch.head '.length)
      else if (m.startsWith('branch.upstream ')) out.upstream = m.slice('branch.upstream '.length)
      else if (m.startsWith('branch.ab ')) {
        const ab = m.slice('branch.ab '.length).split(' ')
        for (const tok of ab) {
          if (tok.startsWith('+')) out.ahead = parseInt(tok.slice(1), 10) || 0
          if (tok.startsWith('-')) out.behind = parseInt(tok.slice(1), 10) || 0
        }
      }
      i += 1
      continue
    }

    if (kind === '1') {
      // Ordinary change: "1 XY sub mH mI mW hH hI path"
      const parts = rec.split(' ')
      const xy = parts[1] || '..'
      const x = xy[0]
      const y = xy[1]
      const filePath = parts.slice(8).join(' ')
      addEntry(out, x, y, filePath)
      i += 1
      continue
    }

    if (kind === '2') {
      // Rename/copy: "2 XY sub mH mI mW hH hI Xscore path" then NUL origPath.
      const parts = rec.split(' ')
      const xy = parts[1] || '..'
      const x = xy[0]
      const y = xy[1]
      const filePath = parts.slice(9).join(' ')
      const origPath = records[i + 1] || ''
      addEntry(out, x, y, filePath, origPath)
      i += 2
      continue
    }

    if (kind === 'u') {
      // Unmerged (conflict): "u XY sub m1 m2 m3 mW h1 h2 h3 path"
      const parts = rec.split(' ')
      const filePath = parts.slice(10).join(' ')
      out.conflicted.push({ path: filePath, status: 'conflicted', staged: false })
      i += 1
      continue
    }

    if (kind === '?') {
      // Untracked: "? path"
      out.untracked.push({ path: rec.slice(2), status: 'untracked', staged: false })
      i += 1
      continue
    }

    if (kind === '!') {
      // Ignored — skip (we don't list ignored files in the SCM panel).
      i += 1
      continue
    }

    i += 1
  }

  return out
}

function addEntry(out, x, y, filePath, origPath) {
  // Staged side (index): X is non-'.'; unstaged side (worktree): Y is non-'.'.
  if (x && x !== '.' && x !== ' ') {
    out.staged.push({ path: filePath, origPath: origPath || undefined, status: statusWord(x, '.'), staged: true })
  }
  if (y && y !== '.' && y !== ' ') {
    out.unstaged.push({ path: filePath, origPath: origPath || undefined, status: statusWord('.', y), staged: false })
  }
}

async function gitStatusForIpc(gitBinary, cwd) {
  const root = resolveRepoRoot(cwd)
  if (!root) return { ok: false, error: 'not-a-repo' }

  const res = await runGit(gitBinary, root, ['status', '--porcelain=v2', '-z', '--branch'])
  if (res.code !== 0) return { ok: false, error: res.stderr.trim() || 'git-status-failed' }

  const parsed = parseStatus(res.stdout)
  return { ok: true, root, ...parsed }
}

// Diff for one path. `staged` true → index vs HEAD; false → worktree vs index.
// Untracked files have no diff target, so we synthesize an all-added diff.
async function gitDiffForIpc(gitBinary, cwd, filePath, staged) {
  const root = resolveRepoRoot(cwd)
  if (!root) return { ok: false, error: 'not-a-repo' }
  if (typeof filePath !== 'string' || !filePath) return { ok: false, error: 'bad-path' }

  const args = ['diff', '--no-color']
  if (staged) args.push('--cached')
  args.push('--', filePath)

  let res = await runGit(gitBinary, root, args)

  // Empty + untracked → show the whole file as additions via /dev/null compare.
  if (res.code === 0 && !res.stdout.trim()) {
    const untracked = await runGit(gitBinary, root, [
      'diff',
      '--no-color',
      '--no-index',
      '--',
      '/dev/null',
      filePath
    ])
    if (untracked.stdout.trim()) res = untracked
  }

  return { ok: true, diff: res.stdout }
}

async function gitStageForIpc(gitBinary, cwd, paths) {
  const root = resolveRepoRoot(cwd)
  if (!root) return { ok: false, error: 'not-a-repo' }
  const list = sanitizePaths(paths)
  if (!list.length) return { ok: false, error: 'no-paths' }

  // `git add` stages adds + modifications + deletions (with --all on pathspec).
  const res = await runGit(gitBinary, root, ['add', '--all', '--', ...list])
  if (res.code !== 0) return { ok: false, error: res.stderr.trim() || 'stage-failed' }
  return { ok: true }
}

async function gitUnstageForIpc(gitBinary, cwd, paths) {
  const root = resolveRepoRoot(cwd)
  if (!root) return { ok: false, error: 'not-a-repo' }
  const list = sanitizePaths(paths)
  if (!list.length) return { ok: false, error: 'no-paths' }

  // `git restore --staged` is the modern unstage; falls back cleanly on old git
  // via `reset` if restore is unavailable.
  let res = await runGit(gitBinary, root, ['restore', '--staged', '--', ...list])
  if (res.code !== 0 && /unknown|not a git command|usage/i.test(res.stderr)) {
    res = await runGit(gitBinary, root, ['reset', '-q', 'HEAD', '--', ...list])
  }
  if (res.code !== 0) return { ok: false, error: res.stderr.trim() || 'unstage-failed' }
  return { ok: true }
}

// Discard worktree changes for tracked paths; delete untracked paths.
async function gitDiscardForIpc(gitBinary, cwd, paths) {
  const root = resolveRepoRoot(cwd)
  if (!root) return { ok: false, error: 'not-a-repo' }
  const list = sanitizePaths(paths)
  if (!list.length) return { ok: false, error: 'no-paths' }

  // restore worktree (tracked) — ignore failures for untracked, then clean them.
  await runGit(gitBinary, root, ['restore', '--worktree', '--', ...list])
  const res = await runGit(gitBinary, root, ['clean', '-fd', '--', ...list])
  if (res.code !== 0) return { ok: false, error: res.stderr.trim() || 'discard-failed' }
  return { ok: true }
}

async function gitCommitForIpc(gitBinary, cwd, message, options = {}) {
  const root = resolveRepoRoot(cwd)
  if (!root) return { ok: false, error: 'not-a-repo' }
  if (typeof message !== 'string' || !message.trim()) return { ok: false, error: 'empty-message' }

  const args = ['commit', '-m', message]
  if (options.amend) args.push('--amend')
  if (options.all) args.push('--all') // stage tracked modifications too

  const res = await runGit(gitBinary, root, args)
  if (res.code !== 0) {
    return { ok: false, error: (res.stderr || res.stdout).trim() || 'commit-failed' }
  }
  return { ok: true, output: res.stdout.trim() }
}

async function gitPushForIpc(gitBinary, cwd, options = {}) {
  const root = resolveRepoRoot(cwd)
  if (!root) return { ok: false, error: 'not-a-repo' }

  const args = ['push']
  if (options.setUpstream) args.push('--set-upstream', 'origin', 'HEAD')

  const res = await runGit(gitBinary, root, args, { timeout: 120_000 })
  if (res.code !== 0) {
    return { ok: false, error: (res.stderr || res.stdout).trim() || 'push-failed' }
  }
  return { ok: true, output: (res.stderr || res.stdout).trim() }
}

async function gitPullForIpc(gitBinary, cwd) {
  const root = resolveRepoRoot(cwd)
  if (!root) return { ok: false, error: 'not-a-repo' }

  const res = await runGit(gitBinary, root, ['pull', '--ff-only'], { timeout: 120_000 })
  if (res.code !== 0) {
    return { ok: false, error: (res.stderr || res.stdout).trim() || 'pull-failed' }
  }
  return { ok: true, output: (res.stdout || res.stderr).trim() }
}

async function gitFetchForIpc(gitBinary, cwd) {
  const root = resolveRepoRoot(cwd)
  if (!root) return { ok: false, error: 'not-a-repo' }

  const res = await runGit(gitBinary, root, ['fetch', '--prune'], { timeout: 120_000 })
  if (res.code !== 0) {
    return { ok: false, error: (res.stderr || res.stdout).trim() || 'fetch-failed' }
  }
  return { ok: true, output: (res.stderr || res.stdout).trim() }
}

// ─── Branches ────────────────────────────────────────────────────────────────

// List local + remote branches via for-each-ref (-z NUL-delimited, scriptable
// format). Returns { ok, current, local[], remote[] } where each branch entry
// is { name, current, upstream, ahead, behind, sha, subject }.
async function gitBranchesForIpc(gitBinary, cwd) {
  const root = resolveRepoRoot(cwd)
  if (!root) return { ok: false, error: 'not-a-repo' }

  // Fields joined by \x1f (unit sep); records by \x00 so subjects/names with
  // spaces survive intact.
  const fmt =
    '%(refname)%1f%(refname:short)%1f%(HEAD)%1f%(upstream:short)%1f%(upstream:track)%1f%(objectname:short)%1f%(contents:subject)%00'
  const res = await runGit(gitBinary, root, [
    'for-each-ref',
    '--format=' + fmt,
    'refs/heads',
    'refs/remotes'
  ])
  if (res.code !== 0) return { ok: false, error: res.stderr.trim() || 'branches-failed' }

  const local = []
  const remote = []
  let current = null

  for (const rec of res.stdout.split('\u0000')) {
    if (!rec.trim()) continue
    const [refname, short, head, upstream, track, sha, subject] = rec.split('\u001f')
    if (refname === 'refs/stash') continue

    // Parse "[ahead 2, behind 1]" style upstream:track tokens.
    let ahead = 0
    let behind = 0
    const aMatch = /ahead (\d+)/.exec(track || '')
    const bMatch = /behind (\d+)/.exec(track || '')
    if (aMatch) ahead = parseInt(aMatch[1], 10) || 0
    if (bMatch) behind = parseInt(bMatch[1], 10) || 0

    const entry = {
      name: short,
      current: head === '*',
      upstream: upstream || null,
      ahead,
      behind,
      sha: sha || '',
      subject: subject || ''
    }

    if (refname.startsWith('refs/remotes/')) {
      // Skip the symbolic "origin/HEAD" pointer.
      if (/\/HEAD$/.test(refname)) continue
      remote.push(entry)
    } else {
      if (entry.current) current = entry.name
      local.push(entry)
    }
  }

  return { ok: true, current, local, remote }
}

async function gitCheckoutForIpc(gitBinary, cwd, branch) {
  const root = resolveRepoRoot(cwd)
  if (!root) return { ok: false, error: 'not-a-repo' }
  if (typeof branch !== 'string' || !branch.trim()) return { ok: false, error: 'bad-branch' }

  // `git switch` is the modern checkout; tracks a remote branch automatically
  // when the name matches a single remote (e.g. switching to origin/feature).
  let res = await runGit(gitBinary, root, ['switch', '--', branch.trim()])
  if (res.code !== 0 && /unknown|not a git command|usage/i.test(res.stderr)) {
    res = await runGit(gitBinary, root, ['checkout', branch.trim()])
  }
  if (res.code !== 0) return { ok: false, error: (res.stderr || res.stdout).trim() || 'checkout-failed' }
  return { ok: true, output: (res.stderr || res.stdout).trim() }
}

async function gitCreateBranchForIpc(gitBinary, cwd, name, options = {}) {
  const root = resolveRepoRoot(cwd)
  if (!root) return { ok: false, error: 'not-a-repo' }
  if (typeof name !== 'string' || !name.trim()) return { ok: false, error: 'bad-branch' }

  // -c creates + switches. A startPoint (sha/branch) is optional.
  const args = ['switch', '-c', name.trim()]
  if (options.startPoint && typeof options.startPoint === 'string') {
    args.push(options.startPoint.trim())
  }

  let res = await runGit(gitBinary, root, args)
  if (res.code !== 0 && /unknown|not a git command|usage/i.test(res.stderr)) {
    const legacy = ['checkout', '-b', name.trim()]
    if (options.startPoint) legacy.push(options.startPoint.trim())
    res = await runGit(gitBinary, root, legacy)
  }
  if (res.code !== 0) return { ok: false, error: (res.stderr || res.stdout).trim() || 'create-branch-failed' }
  return { ok: true, output: (res.stderr || res.stdout).trim() }
}

async function gitDeleteBranchForIpc(gitBinary, cwd, name, options = {}) {
  const root = resolveRepoRoot(cwd)
  if (!root) return { ok: false, error: 'not-a-repo' }
  if (typeof name !== 'string' || !name.trim()) return { ok: false, error: 'bad-branch' }

  const flag = options.force ? '-D' : '-d'
  const res = await runGit(gitBinary, root, ['branch', flag, '--', name.trim()])
  if (res.code !== 0) return { ok: false, error: (res.stderr || res.stdout).trim() || 'delete-branch-failed' }
  return { ok: true, output: res.stdout.trim() }
}

// ─── Log / history ───────────────────────────────────────────────────────────

// Recent commits. Returns { ok, commits[] } with { sha, shortSha, author,
// authorEmail, date (ISO), relativeDate, subject }.
async function gitLogForIpc(gitBinary, cwd, options = {}) {
  const root = resolveRepoRoot(cwd)
  if (!root) return { ok: false, error: 'not-a-repo' }

  const limit = Math.min(Math.max(parseInt(options.limit, 10) || 50, 1), 500)
  // \x1f field sep, \x00 record sep — survives multi-line/odd subjects.
  const fmt = '%H%x1f%h%x1f%an%x1f%ae%x1f%aI%x1f%ar%x1f%s%x00'
  const res = await runGit(gitBinary, root, ['log', `--max-count=${limit}`, '--format=' + fmt])
  if (res.code !== 0) return { ok: false, error: res.stderr.trim() || 'log-failed' }

  const commits = []
  for (const rec of res.stdout.split('\u0000')) {
    if (!rec.trim()) continue
    const [sha, shortSha, author, authorEmail, date, relativeDate, subject] = rec.replace(/^\n/, '').split('\u001f')
    if (!sha) continue
    commits.push({ sha, shortSha, author, authorEmail, date, relativeDate, subject })
  }

  return { ok: true, commits }
}

// Full diff for one commit (against its first parent).
async function gitCommitDiffForIpc(gitBinary, cwd, sha) {
  const root = resolveRepoRoot(cwd)
  if (!root) return { ok: false, error: 'not-a-repo' }
  if (typeof sha !== 'string' || !/^[0-9a-fA-F]{4,64}$/.test(sha.trim())) {
    return { ok: false, error: 'bad-sha' }
  }

  const res = await runGit(gitBinary, root, ['show', '--no-color', '--format=fuller', sha.trim()])
  if (res.code !== 0) return { ok: false, error: res.stderr.trim() || 'commit-diff-failed' }
  return { ok: true, diff: res.stdout }
}

// ─── Stash ───────────────────────────────────────────────────────────────────

async function gitStashListForIpc(gitBinary, cwd) {
  const root = resolveRepoRoot(cwd)
  if (!root) return { ok: false, error: 'not-a-repo' }

  const res = await runGit(gitBinary, root, ['stash', 'list', '--format=%gd%x1f%s%x00'])
  if (res.code !== 0) return { ok: false, error: res.stderr.trim() || 'stash-list-failed' }

  const stashes = []
  for (const rec of res.stdout.split('\u0000')) {
    if (!rec.trim()) continue
    const [ref, subject] = rec.replace(/^\n/, '').split('\u001f')
    if (!ref) continue
    stashes.push({ ref, subject: subject || '' })
  }

  return { ok: true, stashes }
}

async function gitStashPushForIpc(gitBinary, cwd, options = {}) {
  const root = resolveRepoRoot(cwd)
  if (!root) return { ok: false, error: 'not-a-repo' }

  const args = ['stash', 'push']
  if (options.includeUntracked) args.push('--include-untracked')
  if (options.message && typeof options.message === 'string') args.push('-m', options.message)

  const res = await runGit(gitBinary, root, args)
  if (res.code !== 0) return { ok: false, error: (res.stderr || res.stdout).trim() || 'stash-failed' }
  return { ok: true, output: res.stdout.trim() }
}

// Apply/pop/drop a stash by ref. `action` ∈ {apply, pop, drop}.
async function gitStashActionForIpc(gitBinary, cwd, action, ref) {
  const root = resolveRepoRoot(cwd)
  if (!root) return { ok: false, error: 'not-a-repo' }
  if (!['apply', 'pop', 'drop'].includes(action)) return { ok: false, error: 'bad-action' }
  // Stash refs look like "stash@{0}" — validate to keep them off the arg line raw.
  const cleanRef = typeof ref === 'string' && /^stash@\{\d+\}$/.test(ref.trim()) ? ref.trim() : null

  const args = ['stash', action]
  if (cleanRef) args.push(cleanRef)

  const res = await runGit(gitBinary, root, args)
  if (res.code !== 0) return { ok: false, error: (res.stderr || res.stdout).trim() || 'stash-action-failed' }
  return { ok: true, output: res.stdout.trim() }
}

// ─── Hunk-level staging (apply a patch fragment to the index) ────────────────

// Stage or unstage a single diff hunk. The renderer sends the exact unified-diff
// patch text (file header + one @@ hunk). We pipe it to `git apply --cached`
// (stage) or `--cached --reverse` (unstage). Writing the patch to a temp file
// keeps it off the arg line and shell entirely.
async function gitApplyHunkForIpc(gitBinary, cwd, patch, options = {}) {
  const root = resolveRepoRoot(cwd)
  if (!root) return { ok: false, error: 'not-a-repo' }
  if (typeof patch !== 'string' || !patch.trim()) return { ok: false, error: 'empty-patch' }

  const os = require('node:os')
  const fs = require('node:fs')
  const nodePath = require('node:path')

  // git apply is whitespace-sensitive and wants a trailing newline.
  const body = patch.endsWith('\n') ? patch : patch + '\n'
  const tmp = nodePath.join(os.tmpdir(), `hermes-hunk-${Date.now()}-${Math.random().toString(36).slice(2)}.patch`)

  try {
    fs.writeFileSync(tmp, body, 'utf8')
    const args = ['apply', '--cached', '--whitespace=nowarn']
    if (options.reverse) args.push('--reverse')
    args.push(tmp)

    const res = await runGit(gitBinary, root, args)
    if (res.code !== 0) return { ok: false, error: (res.stderr || res.stdout).trim() || 'apply-hunk-failed' }
    return { ok: true }
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) }
  } finally {
    try {
      fs.unlinkSync(tmp)
    } catch {
      // best effort
    }
  }
}

// Strip dangerous / empty entries and option-like leading dashes (defense in
// depth — every call already uses `--` before pathspecs).
function sanitizePaths(paths) {
  if (!Array.isArray(paths)) return []
  const out = []
  for (const p of paths) {
    if (typeof p !== 'string') continue
    const trimmed = p.trim()
    if (!trimmed) continue
    out.push(trimmed)
  }
  return out
}

module.exports = {
  parseStatus,
  statusWord,
  resolveRepoRoot,
  gitStatusForIpc,
  gitDiffForIpc,
  gitStageForIpc,
  gitUnstageForIpc,
  gitDiscardForIpc,
  gitCommitForIpc,
  gitPushForIpc,
  gitPullForIpc,
  gitFetchForIpc,
  gitBranchesForIpc,
  gitCheckoutForIpc,
  gitCreateBranchForIpc,
  gitDeleteBranchForIpc,
  gitLogForIpc,
  gitCommitDiffForIpc,
  gitStashListForIpc,
  gitStashPushForIpc,
  gitStashActionForIpc,
  gitApplyHunkForIpc,
  // exported for tests
  _runGit: runGit
}
