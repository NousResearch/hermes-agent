// Repo-first discovery: walk bounded roots for Git repositories using only
// Node's fs APIs. Electron owns this machine-local capability; the renderer
// supplies the profile-scoped policy from Hermes config.

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

const fsp = fs.promises
const DEFAULT_MAX_DEPTH = 3
const MAX_CONCURRENCY = 32
const JUNK_DIRS = new Set(['Applications', 'Library', 'node_modules', 'site-packages', 'vendor', 'venv'])

export interface RepoScanOptions {
  maxDepth?: number
  enabled?: boolean
  excludePaths?: string[]
}

export interface RepoScanPathOptions {
  homeDir?: string
  platform?: NodeJS.Platform
}

interface NormalizedScanPath {
  key: string
  value: string
}

function pathApiFor(platform: NodeJS.Platform): typeof path.posix | typeof path.win32 {
  return platform === 'win32' ? path.win32 : path.posix
}

export function normalizeRepoScanPath(rawPath: string, options: RepoScanPathOptions = {}): NormalizedScanPath | null {
  const platform = options.platform ?? process.platform
  const homeDir = options.homeDir ?? os.homedir()
  const pathApi = pathApiFor(platform)
  const raw = String(rawPath ?? '').trim()

  if (!raw) {
    return null
  }

  let expanded = raw

  if (raw === '~') {
    expanded = homeDir
  } else if (raw.startsWith('~/') || raw.startsWith('~\\')) {
    expanded = pathApi.join(homeDir, raw.slice(2))
  }

  const absolute = pathApi.isAbsolute(expanded) ? expanded : pathApi.resolve(homeDir, expanded)
  const value = pathApi.normalize(absolute)
  const key = platform === 'win32' ? value.toLocaleLowerCase('en-US') : value

  return { key, value }
}

export function repoScanPathIsWithin(candidate: string, parent: string, options: RepoScanPathOptions = {}): boolean {
  const platform = options.platform ?? process.platform
  const pathApi = pathApiFor(platform)
  const candidatePath = normalizeRepoScanPath(candidate, options)
  const parentPath = normalizeRepoScanPath(parent, options)

  if (!candidatePath || !parentPath) {
    return false
  }

  const relative = pathApi.relative(parentPath.key, candidatePath.key)

  return (
    relative === '' || (relative !== '..' && !relative.startsWith(`..${pathApi.sep}`) && !pathApi.isAbsolute(relative))
  )
}

async function mapLimit<T>(items: T[], limit: number, fn: (item: T) => Promise<void>): Promise<void> {
  let cursor = 0

  async function worker(): Promise<void> {
    while (cursor < items.length) {
      const index = cursor
      cursor += 1
      await fn(items[index])
    }
  }

  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, () => worker()))
}

/**
 * Scan roots for Git repositories. An empty root list preserves the historical
 * home-directory scan. Disabled discovery returns before resolving home or
 * reading the filesystem.
 */
export async function scanGitRepos(roots: string[], options: RepoScanOptions = {}) {
  if (options.enabled === false) {
    return []
  }

  const maxDepthValue = Number(options.maxDepth)
  const maxDepth = Number.isFinite(maxDepthValue) && maxDepthValue >= 0 ? maxDepthValue : DEFAULT_MAX_DEPTH
  const pathOptions: RepoScanPathOptions = {}
  const requestedRoots = Array.isArray(roots) && roots.length > 0 ? roots : [os.homedir()]

  const searchRoots = [
    ...new Map(
      requestedRoots
        .map(root => normalizeRepoScanPath(root, pathOptions))
        .filter((entry): entry is NormalizedScanPath => entry !== null)
        .map(entry => [entry.key, entry.value])
    ).values()
  ]

  const exclusions = (options.excludePaths ?? [])
    .map(excluded => normalizeRepoScanPath(excluded, pathOptions))
    .filter((entry): entry is NormalizedScanPath => entry !== null)

  const found = new Map<string, { root: string; label: string }>()

  function isExcluded(candidate: string): boolean {
    return exclusions.some(excluded => repoScanPathIsWithin(candidate, excluded.value, pathOptions))
  }

  async function walk(dir: string, depth: number): Promise<void> {
    if (depth > maxDepth || isExcluded(dir)) {
      return
    }

    let entries: fs.Dirent[]

    try {
      entries = await fsp.readdir(dir, { withFileTypes: true })
    } catch {
      return
    }

    const gitDir = entries.find(entry => entry.name === '.git' && entry.isDirectory())

    if (gitDir) {
      try {
        await fsp.access(path.join(dir, '.git', 'HEAD'), fs.constants.R_OK)
      } catch {
        return
      }

      const normalized = normalizeRepoScanPath(dir, pathOptions)

      if (normalized) {
        found.set(normalized.key, {
          root: normalized.value,
          label: path.basename(normalized.value) || normalized.value
        })
      }

      return
    }

    const subdirs = entries
      .filter(entry => entry.isDirectory() && !entry.name.startsWith('.') && !JUNK_DIRS.has(entry.name))
      .map(entry => path.join(dir, entry.name))

    await mapLimit(subdirs, MAX_CONCURRENCY, subdir => walk(subdir, depth + 1))
  }

  await mapLimit(searchRoots, MAX_CONCURRENCY, root => walk(root, 0))

  return [...found.values()]
}
