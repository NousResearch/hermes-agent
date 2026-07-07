'use strict'

// Commit-graph backend for the review pane's graph section.
//
// Pure, testable module over the git binary. Every operation runs `git` via
// execFile with an ARGUMENT ARRAY (never a shell string) so commit hashes
// can't inject. The repo root is resolved + hardened through
// resolveRequestedPathForIpc + findGitRoot before any spawn.
//
// Git is spawned with GIT_TERMINAL_PROMPT=0 so a missing credential can't hang
// the process, and GIT_OPTIONAL_LOCKS=0 to avoid index-lock contention.

const { execFile } = require('node:child_process')

const { findGitRoot } = require('./git-root.cjs')
const { resolveRequestedPathForIpc } = require('./hardening.cjs')

const MAX_BUFFER = 16 * 1024 * 1024
const DEFAULT_TIMEOUT = 20_000

function root(cwd) {
  let resolved
  try {
    resolved = resolveRequestedPathForIpc(cwd, { purpose: 'Commit graph' })
  } catch {
    return null
  }
  return findGitRoot(resolved)
}

function run(gitBinary, dir, args) {
  return new Promise(resolve => {
    execFile(
      gitBinary,
      ['-C', dir, ...args],
      {
        cwd: dir,
        timeout: DEFAULT_TIMEOUT,
        maxBuffer: MAX_BUFFER,
        windowsHide: true,
        encoding: 'utf8',
        env: { ...process.env, GIT_TERMINAL_PROMPT: '0', GIT_OPTIONAL_LOCKS: '0' }
      },
      (error, stdout, stderr) => {
        resolve({
          code: error && typeof error.code === 'number' ? error.code : error ? 1 : 0,
          stdout: stdout == null ? '' : stdout,
          stderr: stderr == null ? '' : String(stderr)
        })
      }
    )
  })
}

// Commit history for the graph. When an upstream is configured AND the branch
// has diverged, fetches both local (ahead) and upstream (behind) commits using
// `--left-right HEAD...@{upstream}` so the user can see commits they don't
// have yet. Falls back to a plain `git log HEAD` when there's no upstream or
// no divergence.
async function log(gitBinary, cwd, count = 50) {
  const dir = root(cwd)
  if (!dir) return []

  // Check for divergence first — if upstream has commits we don't, include them
  const divRes = await run(gitBinary, dir, [
    'rev-list', '--left-right', '--count', 'HEAD...@{upstream}'
  ])

  const hasUpstream = divRes.code === 0
  const behind = hasUpstream ? parseInt(divRes.stdout.trim().split(/\s+/)[1] || '0', 10) : 0

  if (hasUpstream && behind > 0) {
    // Fetch upstream commits that aren't in HEAD (the `>` side), then HEAD's history
    const maxBehind = Math.min(behind, 30)
    const upstreamRes = await run(gitBinary, dir, [
      'log', `-n${maxBehind}`, '--left-right', '--pretty=format:%m%x1f%H%x1f%s%x1f%an%x1f%ar%x1f%p',
      'HEAD...@{upstream}'
    ])
    const headRes = await run(gitBinary, dir, [
      'log', `-n${count}`, '--pretty=format:%H%x1f%s%x1f%an%x1f%ar%x1f%p', 'HEAD'
    ])

    const upstream = parseLog(upstreamRes, '>')
    const head = parseLog(headRes)
    // Upstream commits first (they're ahead of us), then our history
    return [...upstream, ...head]
  }

  // No divergence or no upstream — plain linear history
  const res = await run(gitBinary, dir, [
    'log', `-n${count}`, '--pretty=format:%H%x1f%s%x1f%an%x1f%ar%x1f%p'
  ])
  return parseLog(res)
}

function parseLog(res, sideMarker) {
  if (res.code !== 0 || !res.stdout.trim()) return []
  return res.stdout
    .trim()
    .split('\n')
    .map(line => {
      const parts = line.split('\x1f')
      // With %m format, first field is < or >. Without %m, first field is hash.
      const hasMark = parts[0] === '<' || parts[0] === '>'
      const mark = hasMark ? parts[0] : null
      const offset = hasMark ? 1 : 0
      const hash = parts[offset] ?? ''
      const message = parts[offset + 1] ?? ''
      const author = parts[offset + 2] ?? ''
      const date = parts[offset + 3] ?? ''
      const parents = parts[offset + 4] ?? ''
      return { hash, message, author, date, parents: parents ? parents.split(' ').filter(Boolean) : [], mark }
    })
    .filter(e => {
      if (!e.hash || !e.message) return false
      if (!sideMarker) return true
      // When sideMarker='>', keep only upstream commits. When '<', only local.
      return e.mark === sideMarker
    })
    .map(e => {
      const entry = { hash: e.hash, message: e.message, author: e.author, date: e.date, parents: e.parents }
      if (e.mark === '>') entry.side = 'upstream'
      return entry
    })
}

// Files changed in a specific commit (by hash). Returns path + status letter.
// Paths are relative to the REQUESTED cwd (not the git root), so the renderer
// can join cwd + path to get the correct absolute path for preview/diff.
async function show(gitBinary, cwd, hash) {
  const dir = root(cwd)
  if (!dir) return []

  const res = await run(gitBinary, dir, [
    'show', '--no-color', '--name-status', '--pretty=format:', hash
  ])
  if (res.code !== 0 || !res.stdout.trim()) return []

  const { join, relative } = require('node:path')

  return res.stdout
    .trim()
    .split('\n')
    .filter(Boolean)
    .map(line => {
      const parts = line.split('\t')
      const code = parts[0].charAt(0)
      const filePath = parts.length > 2 ? parts[parts.length - 1] : parts[1] ?? ''
      // git show returns paths relative to the repo root (dir). Re-base them
      // relative to the requested cwd so the renderer's absolutePath() works.
      return { path: relative(cwd, join(dir, filePath.trim())), status: code }
    })
    .filter(f => f.path)
}

// Ahead/behind count vs upstream. Returns { ahead, behind } or null if no
// upstream is configured (branch not pushed / no tracking remote).
async function divergence(gitBinary, cwd) {
  const dir = root(cwd)
  if (!dir) return null

  const res = await run(gitBinary, dir, [
    'rev-list', '--left-right', '--count', 'HEAD...@{upstream}'
  ])
  if (res.code !== 0) return null

  const [ahead, behind] = res.stdout.trim().split(/\s+/).map(Number)
  return { ahead: ahead || 0, behind: behind || 0 }
}

module.exports = { log, show, divergence }
