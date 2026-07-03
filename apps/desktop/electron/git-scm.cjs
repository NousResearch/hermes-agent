'use strict'

// Source-control backend for the desktop Source Control panel.
//
// Pure, testable module over the git binary. Every operation runs `git` via
// execFile with an ARGUMENT ARRAY (never a shell string) so file names /
// commit messages can't inject. The repo root is resolved + hardened through
// resolveRequestedPathForIpc before any spawn.
//
// Status uses porcelain=v2 + -z (NUL-delimited) so filenames with spaces,
// quotes, newlines, and unicode parse unambiguously.
//
// Git is spawned with GIT_TERMINAL_PROMPT=0 so a missing credential can't hang
// the process, and GIT_OPTIONAL_LOCKS=0 to avoid index-lock contention.

const { execFile } = require('node:child_process')

const { findGitRoot } = require('./git-root.cjs')
const { resolveRequestedPathForIpc } = require('./hardening.cjs')

const MAX_BUFFER = 16 * 1024 * 1024
const DEFAULT_TIMEOUT = 20_000

// Resolve a renderer-supplied cwd to a hardened absolute git root, or null.
function root(cwd) {
  let resolved
  try {
    resolved = resolveRequestedPathForIpc(cwd, { purpose: 'Git source control' })
  } catch {
    return null
  }
  return findGitRoot(resolved)
}

// Run git with an arg array in `dir`. Never rejects — callers decide what
// a non-zero exit means (e.g. `git diff` exits 1 when there are differences).
function run(gitBinary, dir, args, opts = {}) {
  return new Promise(resolve => {
    execFile(
      gitBinary,
      ['-C', dir, ...args],
      {
        cwd: dir,
        timeout: opts.timeout || DEFAULT_TIMEOUT,
        maxBuffer: MAX_BUFFER,
        windowsHide: true,
        encoding: 'utf8',
        env: { ...process.env, GIT_TERMINAL_PROMPT: '0', GIT_OPTIONAL_LOCKS: '0' }
      },
      (error, stdout, stderr) => {
        resolve({
          code: error && typeof error.code === 'number' ? error.code : error ? 1 : 0,
          stdout: stdout == null ? '' : stdout,
          stderr: stderr == null ? '' : String(stderr),
          timedOut: Boolean(error && error.killed)
        })
      }
    )
  })
}

// ── Status (porcelain v2) ───────────────────────────────────────────────

// XY status code → friendly word the UI renders.
function statusWord(x, y) {
  const code = (x || ' ') + (y || ' ')
  if (code.includes('U') || x === 'U' || y === 'U') return 'conflicted'
  const c = x !== '.' && x !== ' ' ? x : y
  switch (c) {
    case 'M': return 'modified'
    case 'A': return 'added'
    case 'D': return 'deleted'
    case 'R': return 'renamed'
    case 'C': return 'copied'
    case 'T': return 'typechange'
    default: return 'modified'
  }
}

// Parse `git status --porcelain=v2 -z --branch` into structured groups.
function parseStatus(z) {
  const out = {
    branch: null, upstream: null, ahead: 0, behind: 0,
    staged: [], unstaged: [], untracked: [], conflicted: []
  }
  const records = z.split('\u0000')
  let i = 0

  while (i < records.length) {
    const rec = records[i]
    if (!rec) { i += 1; continue }

    const kind = rec[0]

    if (kind === '#') {
      const m = rec.slice(2)
      if (m.startsWith('branch.head ')) out.branch = m.slice('branch.head '.length)
      else if (m.startsWith('branch.upstream ')) out.upstream = m.slice('branch.upstream '.length)
      else if (m.startsWith('branch.ab ')) {
        for (const tok of m.slice('branch.ab '.length).split(' ')) {
          if (tok.startsWith('+')) out.ahead = parseInt(tok.slice(1), 10) || 0
          if (tok.startsWith('-')) out.behind = parseInt(tok.slice(1), 10) || 0
        }
      }
      i += 1; continue
    }

    if (kind === '1') {
      const parts = rec.split(' ')
      const xy = parts[1] || '..'
      addEntry(out, xy[0], xy[1], parts.slice(8).join(' '))
      i += 1; continue
    }

    if (kind === '2') {
      const parts = rec.split(' ')
      const xy = parts[1] || '..'
      const filePath = parts.slice(9).join(' ')
      const origPath = records[i + 1] || ''
      addEntry(out, xy[0], xy[1], filePath, origPath)
      i += 2; continue
    }

    if (kind === 'u') {
      const parts = rec.split(' ')
      out.conflicted.push({ path: parts.slice(10).join(' '), status: 'conflicted', staged: false })
      i += 1; continue
    }

    if (kind === '?') {
      out.untracked.push({ path: rec.slice(2), status: 'untracked', staged: false })
      i += 1; continue
    }

    i += 1
  }

  return out
}

function addEntry(out, x, y, filePath, origPath) {
  if (x && x !== '.' && x !== ' ') {
    out.staged.push({ path: filePath, origPath: origPath || undefined, status: statusWord(x, '.'), staged: true })
  }
  if (y && y !== '.' && y !== ' ') {
    out.unstaged.push({ path: filePath, origPath: origPath || undefined, status: statusWord('.', y), staged: false })
  }
}

// ── IPC-shaped operations ────────────────────────────────────────────────

// Repo status for the REPOSITORY section. Returns HermesRepoStatus-shaped
// object (compatible with the existing repoStatus() in git-review-ops.cjs).
// Uses execFile + porcelain v2 instead of simple-git to avoid the Windows
// binary path crash and give us security control.
async function status(gitBinary, cwd) {
  const dir = root(cwd)
  if (!dir) return null

  const res = await run(gitBinary, dir, ['status', '--porcelain=v2', '-z', '--branch'])
  if (res.code !== 0) return null

  const parsed = parseStatus(res.stdout)
  const detached = parsed.branch === '(detached)'

  // Count added/removed lines vs HEAD (best-effort).
  let added = 0
  let removed = 0
  const diff = await run(gitBinary, dir, ['diff', '--numstat', 'HEAD'])
  if (diff.code === 0) {
    for (const line of diff.stdout.trim().split('\n').filter(Boolean)) {
      const [a, r] = line.split('\t')
      added += parseInt(a, 10) || 0
      removed += parseInt(r, 10) || 0
    }
  }

  // Detect default branch.
  let defaultBranch = null
  const head = await run(gitBinary, dir, ['symbolic-ref', '--short', 'refs/remotes/origin/HEAD'])
  if (head.code === 0) {
    defaultBranch = head.stdout.trim().replace(/^origin\//, '') || null
  }

  const files = [
    ...parsed.staged.map(f => ({ path: f.path, staged: true, unstaged: false, untracked: false, conflicted: false })),
    ...parsed.unstaged.map(f => ({ path: f.path, staged: false, unstaged: true, untracked: false, conflicted: false })),
    ...parsed.untracked.map(f => ({ path: f.path, staged: false, unstaged: false, untracked: true, conflicted: false })),
    ...parsed.conflicted.map(f => ({ path: f.path, staged: false, unstaged: false, untracked: false, conflicted: true }))
  ]

  return {
    branch: detached ? null : parsed.branch,
    defaultBranch,
    detached,
    ahead: parsed.ahead,
    behind: parsed.behind,
    staged: parsed.staged.length,
    unstaged: parsed.unstaged.length,
    untracked: parsed.untracked.length,
    conflicted: parsed.conflicted.length,
    changed: files.length,
    added,
    removed,
    files: files.slice(0, 200)
  }
}

// Changed files (staged + unstaged + untracked) for the CHANGES section.
// Returns HermesReviewFile-shaped entries so the UI doesn't need a separate type.
async function changedFiles(gitBinary, cwd) {
  const dir = root(cwd)
  if (!dir) return { files: [], base: null }

  const res = await run(gitBinary, dir, ['status', '--porcelain=v2', '-z', '--branch'])
  if (res.code !== 0) return { files: [], base: null }

  const parsed = parseStatus(res.stdout)
  const files = []

  for (const f of parsed.staged) {
    files.push({ path: f.path, status: statusLetter(f.status), staged: true, added: 0, removed: 0 })
  }
  for (const f of parsed.unstaged) {
    files.push({ path: f.path, status: statusLetter(f.status), staged: false, added: 0, removed: 0 })
  }
  for (const f of parsed.untracked) {
    files.push({ path: f.path, status: '?', staged: false, added: 0, removed: 0 })
  }

  // Per-file line counts (best-effort, bounded).
  const tracked = files.filter(f => f.status !== '?')
  for (let i = 0; i < tracked.length; i += 20) {
    const batch = tracked.slice(i, i + 20)
    await Promise.all(batch.map(async f => {
      const d = await run(gitBinary, dir, ['diff', '--numstat', '--', f.path])
      if (d.code === 0) {
        const [a, r] = d.stdout.trim().split('\t')
        f.added = parseInt(a, 10) || 0
        f.removed = parseInt(r, 10) || 0
      }
    }))
  }

  return { files, base: null }
}

// Map statusWord → single-letter code for the UI badge.
function statusLetter(word) {
  switch (word) {
    case 'added': return 'A'
    case 'deleted': return 'D'
    case 'renamed': return 'R'
    case 'copied': return 'C'
    case 'typechange': return 'T'
    case 'conflicted': return 'U'
    default: return 'M'
  }
}

// Commit history for the GRAPH section. Returns last N commits with hash,
// message, author, date, and parents.
async function log(gitBinary, cwd, count = 50) {
  const dir = root(cwd)
  if (!dir) return []

  const res = await run(gitBinary, dir, [
    'log', `-n${count}`, '--pretty=format:%H%x1f%s%x1f%an%x1f%ar%x1f%p'
  ])
  if (res.code !== 0 || !res.stdout.trim()) return []

  return res.stdout
    .trim()
    .split('\n')
    .map(line => {
      const [hash, message, author, date, parents] = line.split('\x1f')
      return { hash, message, author, date, parents: parents ? parents.split(' ').filter(Boolean) : [] }
    })
    .filter(e => e.hash && e.message)
}

// Files changed in a specific commit (hash). Returns path + status letter.
async function show(gitBinary, cwd, hash) {
  const dir = root(cwd)
  if (!dir) return []

  const res = await run(gitBinary, dir, [
    'show', '--no-color', '--name-status', '--pretty=format:', hash
  ])
  if (res.code !== 0 || !res.stdout.trim()) return []

  return res.stdout
    .trim()
    .split('\n')
    .filter(Boolean)
    .map(line => {
      const parts = line.split('\t')
      const code = parts[0].charAt(0)
      // R100\told\tnew → take last element; M\tpath → take element[1]
      const filePath = parts.length > 2 ? parts[parts.length - 1] : parts[1] ?? ''
      return { path: filePath.trim(), status: code }
    })
    .filter(f => f.path)
}

// Working-tree-vs-HEAD unified diff for one file. Returns empty string on
// no changes. For untracked files, synthesizes an all-added diff via
// --no-index /dev/null comparison.
async function fileDiff(gitBinary, cwd, filePath) {
  const dir = root(cwd)
  if (!dir) return ''

  const res = await run(gitBinary, dir, ['diff', '--no-color', 'HEAD', '--', filePath])
  if (res.stdout.trim()) return res.stdout

  // No tracked changes vs HEAD. Check if it's untracked.
  const st = await run(gitBinary, dir, ['status', '--porcelain', '--', filePath])
  if (!st.stdout.trim().startsWith('??')) return ''

  // Synthesize all-added diff for untracked files.
  const synth = await run(gitBinary, dir, ['diff', '--no-index', '--', '/dev/null', filePath])
  return synth.stdout
}

module.exports = {
  parseStatus,
  statusWord,
  status,
  changedFiles,
  log,
  show,
  fileDiff,
  // exported for tests
  _run: run,
  _root: root
}
