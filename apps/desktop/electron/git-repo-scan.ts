// Repo-first discovery: walk bounded roots for git repos using only Node's `fs`
// — no native addon, so it just works for anyone who pulls main (no
// electron-rebuild). Mirrors how GitHub Desktop scans: stop at the first `.git`
// (don't descend into a repo), cap depth, and skip heavy non-repo trees so the
// first scan stays fast. Results are cached by the backend after the first run.

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

const fsp = fs.promises

// Shallow on purpose: real projects live a few levels under home
// (`~/www/repo`, `~/code/org/repo`); deeper `.git` dirs are almost always
// fixtures/vendored/eval checkouts (e.g. `~/www/ha-evals/tasks/*/repo`). Repos
// you actually use but keep deeper still surface via session-derived discovery,
// so this only prunes noise, never repos with history.
const DEFAULT_MAX_DEPTH = 3
const MAX_CONCURRENCY = 32

// Big trees that are never themselves repos and would waste the walk. Anything
// hidden (dotdirs like .cache/.Trash/.npm) is skipped wholesale below, so this
// only needs the non-hidden heavyweights.
const JUNK_DIRS = new Set(['Applications', 'Library', 'node_modules', 'site-packages', 'vendor', 'venv'])

// Avoid macOS TCC prompts when the default home scan reaches protected media
// folders. Nested names and explicitly supplied media roots remain scannable.
const MEDIA_ROOT_DIRS = new Set(['Movies', 'Music', 'Pictures', 'Public'])

// These packages look like directories but are TCC-protected and cannot contain
// user repositories, including when stored outside the default media folders.
const LIBRARY_PACKAGE_SUFFIXES = ['.photoslibrary', '.musiclibrary', '.tvlibrary', '.aplibrary']

function isLibraryPackage(name) {
  const lower = String(name).toLowerCase()
  return LIBRARY_PACKAGE_SUFFIXES.some(suffix => lower.endsWith(suffix))
}

async function mapLimit(items, limit, fn) {
  let cursor = 0

  async function worker() {
    while (cursor < items.length) {
      const index = cursor
      cursor += 1
      await fn(items[index])
    }
  }

  await Promise.all(Array.from({ length: Math.min(limit, items.length) } as any, worker))
}

/**
 * Scan `roots` (default: the home dir) for git repositories. Returns deduped
 * `{ root, label }` entries. `options.maxDepth` caps recursion (default 3).
 */
async function scanGitRepos(roots, options: any = {}) {
  const maxDepth = Number(options.maxDepth) || DEFAULT_MAX_DEPTH
  const searchRoots = Array.isArray(roots) && roots.length > 0 ? roots : [os.homedir()]
  const skipTccProtectedPaths = process.platform === 'darwin'
  const found = new Map()

  async function walk(dir, depth) {
    if (depth > maxDepth) {
      return
    }

    let entries

    try {
      entries = await fsp.readdir(dir, { withFileTypes: true })
    } catch {
      return // unreadable / permission denied
    }

    // A `.git` DIRECTORY marks a real repo root (a main checkout). A `.git`
    // FILE is a linked worktree or submodule — those belong to their parent
    // repo as lanes, not as separate projects, so we don't list them (and we
    // keep descending in case a real repo sits deeper). This is what kills the
    // worktree/eval-repo duplicate explosion.
    if (entries.some(entry => entry.name === '.git' && entry.isDirectory())) {
      const root = dir.replace(/[/\\]+$/, '')
      found.set(root, path.basename(root) || root)

      return
    }

    const subdirs = []

    for (const entry of entries) {
      // Real directories only (skip symlinks to avoid loops), no hidden dirs, no
      // known heavy trees.
      if (!entry.isDirectory() || entry.name.startsWith('.') || JUNK_DIRS.has(entry.name)) {
        continue
      }

      if (skipTccProtectedPaths && depth === 0 && MEDIA_ROOT_DIRS.has(entry.name)) {
        continue
      }

      if (skipTccProtectedPaths && isLibraryPackage(entry.name)) {
        continue
      }

      subdirs.push(path.join(dir, entry.name))
    }

    await mapLimit(subdirs, MAX_CONCURRENCY, sub => walk(sub, depth + 1))
  }

  await mapLimit(searchRoots.map(root => String(root || '').trim()).filter(Boolean), MAX_CONCURRENCY, root =>
    walk(root, 0)
  )

  return [...found.entries()].map(([root, label]) => ({ label, root }))
}

export { scanGitRepos }
