import ignore from 'ignore'

import type { HermesReadDirEntry, HermesReadDirResult } from '@/global'
import { desktopFsCacheKey, desktopGitRoot, readDesktopDir, readDesktopFileDataUrl } from '@/lib/desktop-fs'
import { ALWAYS_EXCLUDED } from '@/lib/excluded-paths'
import { cleanPath, comparisonPath } from '@/lib/path-compare'

export type ProjectTreeEntry = HermesReadDirEntry

interface GitignoreRule {
  base: string
  ig: ReturnType<typeof ignore>
}

const gitRootCache = new Map<string, Promise<string | null>>()
const gitignoreCache = new Map<string, Promise<GitignoreRule | null>>()
const nestedRepoCache = new Map<string, Promise<boolean>>()

function decodeDataUrl(dataUrl: string) {
  const match = dataUrl.match(/^data:[^,]*,(.*)$/)
  const data = match?.[1] || ''
  const isBase64 = dataUrl.slice(0, dataUrl.indexOf(',')).includes(';base64')

  if (!isBase64) {
    return decodeURIComponent(data)
  }

  const bytes = Uint8Array.from(atob(data), ch => ch.charCodeAt(0))

  return new TextDecoder().decode(bytes)
}

/** Strict POSIX-style relative path; null if `child` is not inside `root`. */
function relativeTo(root: string, child: string) {
  const r = cleanPath(root)
  const c = cleanPath(child)
  const rKey = comparisonPath(r)
  const cKey = comparisonPath(c)

  if (cKey === rKey) {
    return ''
  }

  return cKey.startsWith(`${rKey}/`) ? c.slice(r.length + 1) : null
}

/** Repo-root → repo-root/a → repo-root/a/b → … for every dir between root and `dir`. */
function ancestorDirs(root: string, dir: string) {
  const r = cleanPath(root)
  const rel = relativeTo(r, dir)

  if (rel === null || rel === '') {
    return [r]
  }

  const dirs = [r]
  let current = r

  for (const part of rel.split('/').filter(Boolean)) {
    current = `${current}/${part}`
    dirs.push(current)
  }

  return dirs
}

async function gitRootFor(start: string) {
  const key = `${desktopFsCacheKey()}:${cleanPath(start)}`
  let cached = gitRootCache.get(key)

  if (!cached) {
    cached = desktopGitRoot(cleanPath(start))
    gitRootCache.set(key, cached)
  }

  return cached
}

/** Read .gitignore at `dir` if it actually exists — never probe missing files. */
async function readGitignore(dir: string): Promise<GitignoreRule | null> {
  try {
    const listing = await readDesktopDir(dir)

    if (!listing.entries.some(e => e.name === '.gitignore' && !e.isDirectory)) {
      return null
    }

    const text = decodeDataUrl(await readDesktopFileDataUrl(`${dir}/.gitignore`))

    return { base: dir, ig: ignore().add(text) }
  } catch {
    return null
  }
}

async function gitignoreFor(dir: string) {
  const key = `${desktopFsCacheKey()}:${cleanPath(dir)}`
  let cached = gitignoreCache.get(key)

  if (!cached) {
    cached = readGitignore(cleanPath(dir))
    gitignoreCache.set(key, cached)
  }

  return cached
}

async function isNestedRepoRoot(entry: HermesReadDirEntry): Promise<boolean> {
  if (!entry.isDirectory) {
    return false
  }

  const key = `${desktopFsCacheKey()}:${cleanPath(entry.path)}`
  let cached = nestedRepoCache.get(key)

  if (!cached) {
    cached = (async () => {
      try {
        const listing = await readDesktopDir(entry.path)

        return listing.entries.some(e => e.name === '.git')
      } catch {
        return false
      }
    })()
    nestedRepoCache.set(key, cached)
  }

  return cached
}

function ignoredBy(rules: GitignoreRule[], entry: HermesReadDirEntry) {
  return rules.some(rule => {
    const rel = relativeTo(rule.base, entry.path)

    if (rel === null || rel === '') {
      return false
    }

    return rule.ig.ignores(entry.isDirectory ? `${rel}/` : rel)
  })
}

async function filterIgnored(entries: HermesReadDirEntry[], rootPath: string, dirPath: string) {
  const root = await gitRootFor(rootPath)

  if (!root) {
    return entries
  }

  const rules = (await Promise.all(ancestorDirs(root, dirPath).map(gitignoreFor))).filter((r): r is GitignoreRule =>
    Boolean(r)
  )

  if (rules.length === 0) {
    return entries
  }

  const visible = await Promise.all(
    entries.map(async entry => {
      if (!ignoredBy(rules, entry)) {
        return entry
      }

      return (await isNestedRepoRoot(entry)) ? entry : null
    })
  )

  return visible.filter((entry): entry is HermesReadDirEntry => entry !== null)
}

export async function readProjectDir(dirPath: string, rootPath = dirPath): Promise<HermesReadDirResult> {
  if (!window.hermesDesktop) {
    return { entries: [], error: 'no-bridge' }
  }

  const result = await readDesktopDir(dirPath)
  const entries = (result?.entries ?? []).filter(entry => !ALWAYS_EXCLUDED.has(entry.name))

  return { ...result, entries: await filterIgnored(entries, rootPath, dirPath) }
}

export function clearProjectDirCache(rootPath?: string) {
  if (!rootPath) {
    gitRootCache.clear()
    gitignoreCache.clear()
    nestedRepoCache.clear()

    return
  }

  const normalizedRoot = cleanPath(rootPath)
  const key = `${desktopFsCacheKey()}:${normalizedRoot}`
  gitRootCache.delete(key)
  gitignoreCache.delete(key)

  const nestedPrefix = normalizedRoot === '/' ? key : `${key}/`

  for (const nestedKey of nestedRepoCache.keys()) {
    if (nestedKey === key || nestedKey.startsWith(nestedPrefix)) {
      nestedRepoCache.delete(nestedKey)
    }
  }
}
