'use strict'

// Resolve git-worktree relationships for a set of session cwds, reading git's
// on-disk metadata directly (no `git` spawn per path):
//
//   - A normal checkout has a `.git` DIRECTORY at its root → it's the main
//     worktree; its repo root IS that directory's parent.
//   - A linked worktree has a `.git` FILE: `gitdir: <repo>/.git/worktrees/<name>`.
//     That admin dir's `commondir` points back at the shared `<repo>/.git`, whose
//     parent is the main repo root.
//
// Grouping by repoRoot therefore clusters a repo's main checkout with all of its
// linked worktrees, regardless of how the worktree directories are named. The
// branch (read from the worktree's own HEAD) gives each worktree a meaningful
// label.

const fs = require('node:fs')
const path = require('node:path')
const { resolveRequestedPathForIpc } = require('./hardening.cjs')

// Walk up from `start` to the nearest ancestor that carries a `.git` entry
// (file for a linked worktree, dir for the main checkout). Capped so a stray
// path can't loop forever.
function findGitHost(start, fsImpl) {
  let dir = start

  for (let i = 0; i < 64; i += 1) {
    const dotgit = path.join(dir, '.git')

    try {
      if (fsImpl.existsSync(dotgit)) {
        return dir
      }
    } catch {
      return null
    }

    const parent = path.dirname(dir)

    if (parent === dir) {
      return null
    }

    dir = parent
  }

  return null
}

function readBranch(gitDir, fsImpl) {
  try {
    const head = fsImpl.readFileSync(path.join(gitDir, 'HEAD'), 'utf8').trim()
    const ref = head.match(/^ref:\s*refs\/heads\/(.+)$/)

    if (ref) {
      return ref[1]
    }

    // Detached HEAD: surface a short sha so the worktree still gets a label.
    return /^[0-9a-f]{7,40}$/i.test(head) ? head.slice(0, 8) : null
  } catch {
    return null
  }
}

// Given the directory that owns the `.git` entry, resolve its worktree identity.
function resolveFromHost(host, fsImpl) {
  const dotgit = path.join(host, '.git')
  let stat

  try {
    stat = fsImpl.statSync(dotgit)
  } catch {
    return null
  }

  if (stat.isDirectory()) {
    return {
      repoRoot: host,
      worktreeRoot: host,
      isMainWorktree: true,
      branch: readBranch(dotgit, fsImpl)
    }
  }

  // Linked worktree: `.git` is a file pointing at the admin dir.
  let contents

  try {
    contents = fsImpl.readFileSync(dotgit, 'utf8').trim()
  } catch {
    return null
  }

  const match = contents.match(/^gitdir:\s*(.+)$/m)

  if (!match) {
    return null
  }

  const adminDir = path.resolve(host, match[1].trim())

  // `commondir` resolves to the shared `<repo>/.git`; fall back to walking two
  // levels up from `<repo>/.git/worktrees/<name>` if it's missing.
  let commonDir

  try {
    const rel = fsImpl.readFileSync(path.join(adminDir, 'commondir'), 'utf8').trim()
    commonDir = path.resolve(adminDir, rel)
  } catch {
    commonDir = path.dirname(path.dirname(adminDir))
  }

  return {
    repoRoot: path.dirname(commonDir),
    worktreeRoot: host,
    isMainWorktree: false,
    branch: readBranch(adminDir, fsImpl)
  }
}

function resolveWorktree(startPath, fsImpl = fs) {
  let resolved

  try {
    resolved = resolveRequestedPathForIpc(startPath, { purpose: 'Worktree lookup' })
  } catch {
    return null
  }

  let start = resolved

  try {
    const stat = fsImpl.statSync(resolved)

    if (!stat.isDirectory()) {
      start = path.dirname(resolved)
    }
  } catch {
    return null
  }

  const host = findGitHost(start, fsImpl)

  if (!host) {
    return null
  }

  return resolveFromHost(host, fsImpl)
}

// Batch entry point for the renderer: maps each requested cwd to its worktree
// info (or null when it isn't inside a git checkout / can't be read). Dedupes so
// many sessions sharing a cwd cost one lookup.
async function worktreesForIpc(cwds, options = {}) {
  const fsImpl = options.fs || fs
  const list = Array.isArray(cwds) ? cwds : []
  const out = {}

  for (const cwd of list) {
    if (typeof cwd !== 'string' || !cwd.trim() || cwd in out) {
      continue
    }

    out[cwd] = resolveWorktree(cwd, fsImpl)
  }

  return out
}

module.exports = {
  resolveWorktree,
  worktreesForIpc
}
