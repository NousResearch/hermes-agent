import ignore from 'ignore'

import type { HermesReadDirEntry, HermesReadDirResult } from '@/global'

export type ProjectTreeEntry = HermesReadDirEntry

interface GitignoreRule {
  base: string
  ig: ReturnType<typeof ignore>
}

const gitRootCache = new Map<string, Promise<string | null>>()
const gitignoreCache = new Map<string, Promise<GitignoreRule | null>>()

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

function clean(path: string) {
  return path.replace(/\/+$/, '') || '/'
}

/** Strict POSIX-style relative path; null if `child` is not inside `root`. */
function relativeTo(root: string, child: string) {
  const r = clean(root)
  const c = clean(child)

  if (c === r) {
    return ''
  }

  return c.startsWith(`${r}/`) ? c.slice(r.length + 1) : null
}

/** Repo-root → repo-root/a → repo-root/a/b → … for every dir between root and `dir`. */
function ancestorDirs(root: string, dir: string) {
  const r = clean(root)
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
  if (!window.hermesDesktop?.gitRoot) {
    return null
  }

  const key = clean(start)
  let cached = gitRootCache.get(key)

  if (!cached) {
    cached = window.hermesDesktop.gitRoot(key)
    gitRootCache.set(key, cached)
  }

  return cached
}

/** Read .gitignore at `dir` if it actually exists — never probe missing files. */
async function readGitignore(dir: string): Promise<GitignoreRule | null> {
  if (!window.hermesDesktop?.readDir || !window.hermesDesktop.readFileDataUrl) {
    return null
  }

  try {
    const listing = await window.hermesDesktop.readDir(dir)

    if (!listing.entries.some(e => e.name === '.gitignore' && !e.isDirectory)) {
      return null
    }

    const text = decodeDataUrl(await window.hermesDesktop.readFileDataUrl(`${dir}/.gitignore`))

    return { base: dir, ig: ignore().add(text) }
  } catch {
    return null
  }
}

async function gitignoreFor(dir: string) {
  const key = clean(dir)
  let cached = gitignoreCache.get(key)

  if (!cached) {
    cached = readGitignore(key)
    gitignoreCache.set(key, cached)
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

  return rules.length > 0 ? entries.filter(entry => !ignoredBy(rules, entry)) : entries
}

export async function readProjectDir(dirPath: string, rootPath = dirPath): Promise<HermesReadDirResult> {
  if (!window.hermesDesktop) {
    return { entries: [], error: 'no-bridge' }
  }

  const result = await window.hermesDesktop.readDir(dirPath)

  return { ...result, entries: await filterIgnored(result.entries, rootPath, dirPath) }
}

export function clearProjectDirCache(rootPath?: string) {
  if (!rootPath) {
    gitRootCache.clear()
    gitignoreCache.clear()

    return
  }

  const key = clean(rootPath)
  gitRootCache.delete(key)
  gitignoreCache.delete(key)
}
