'use strict'

/**
 * install-mutation-set.ts
 *
 * The set of files a Windows update will replace or delete inside a Hermes
 * install: every native module, DLL, and executable under `venv\`.
 *
 * `.hermes-runtime\python\<generation>` is deliberately NOT part of the set:
 * the updater never rewrites a generation in place (a new interpreter lands
 * in a new generation directory), and unrelated tools that borrowed the
 * managed interpreter as their base Python (uv tool venvs such as an MCP
 * server spawned by another agent) map its DLLs without touching the venv.
 * Counting them would block updates on processes the update does not affect.
 *
 * This is the resource the updater actually needs free. The previous proxy —
 * three shim files under `venv\Scripts` — only proved that the uv launcher was
 * gone: the real interpreter runs from `.hermes-runtime` and keeps
 * `venv\Lib\site-packages\**\*.pyd` mapped, and Restart Manager on the shim
 * never listed it (docs/analysis/2026-09-02-windows-update-holder-ownership-gap.md).
 *
 * Pure module: every filesystem call is injectable so the logic is testable
 * without a real install.
 */

import fs from 'node:fs'
import path from 'node:path'

export const INSTALL_MUTATION_EXTENSIONS: ReadonlySet<string> = new Set(['.pyd', '.dll', '.exe'])

/** Enumeration is filesystem-bound (~1 s on a full install); cache it briefly. */
export const INSTALL_MUTATION_SET_TTL_MS = 30_000

export interface MutationSetFs {
  readdirSync: (dir: string, options: { withFileTypes: true }) => fs.Dirent[]
  existsSync: (target: string) => boolean
}

export interface LockProbeFs {
  openSync: (target: string, flags: string) => number
  closeSync: (fd: number) => void
  statSync: (target: string) => { nlink: number }
}

/** Shim files first so a caller that needs one representative resource gets the shim. */
export function installShimCandidates(updateRoot: string): string[] {
  return [
    path.join(updateRoot, 'venv', 'Scripts', 'hermes.exe'),
    path.join(updateRoot, 'venv', 'Scripts', 'python.exe'),
    path.join(updateRoot, 'venv', 'python.exe')
  ]
}

export function installMutationRoots(updateRoot: string): string[] {
  return [path.join(updateRoot, 'venv')]
}

function walk(dir: string, out: string[], fsImpl: MutationSetFs, depth: number): void {
  // Deep enough for venv\Lib\site-packages\<pkg>\<sub>\<sub>\...
  if (depth > 12) {return}

  let entries: fs.Dirent[]

  try {
    entries = fsImpl.readdirSync(dir, { withFileTypes: true })
  } catch {
    return
  }

  for (const entry of entries) {
    const target = path.join(dir, entry.name)

    if (entry.isDirectory()) {
      walk(target, out, fsImpl, depth + 1)
    } else if (entry.isFile() && INSTALL_MUTATION_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) {
      out.push(target)
    }
  }
}

/**
 * Enumerate the mutation set. Shim files come first (when present), then
 * every other matching file in directory order. Missing roots are skipped.
 */
export function enumerateInstallMutationSet(
  updateRoot: string,
  fsImpl: MutationSetFs = fs
): string[] {
  const ordered: string[] = []
  const seen = new Set<string>()

  const add = (target: string) => {
    const key = process.platform === 'win32' ? target.toLowerCase() : target

    if (seen.has(key)) {return}
    seen.add(key)
    ordered.push(target)
  }

  for (const shim of installShimCandidates(updateRoot)) {
    let present = false

    try {
      present = fsImpl.existsSync(shim)
    } catch {
      present = false
    }

    if (present) {add(shim)}
  }

  const found: string[] = []

  for (const root of installMutationRoots(updateRoot)) {
    walk(root, found, fsImpl, 0)
  }

  for (const target of found) {
    add(target)
  }

  return ordered
}

const cache = new Map<string, { at: number; files: string[] }>()

/** Cached enumeration keyed by canonical root. */
export function getInstallMutationSet(
  updateRoot: string,
  options: { now?: () => number; ttlMs?: number; fsImpl?: MutationSetFs } = {}
): string[] {
  const now = options.now ?? Date.now
  const ttlMs = options.ttlMs ?? INSTALL_MUTATION_SET_TTL_MS
  const key = process.platform === 'win32' ? path.resolve(updateRoot).toLowerCase() : path.resolve(updateRoot)
  const hit = cache.get(key)

  if (hit && now() - hit.at < ttlMs) {
    return hit.files
  }

  const files = enumerateInstallMutationSet(updateRoot, options.fsImpl)
  cache.set(key, { at: now(), files })

  return files
}

export function clearInstallMutationSetCache(): void {
  cache.clear()
}

export interface InstallResourceLocks {
  /**
   * Locked files with a single hard link. Only a process holding *our* link
   * can lock these, so each one is a proven blocker of the venv sync.
   */
  definite: string[]
  /**
   * Locked files with more than one hard link. uv links identical wheel files
   * from its cache into every venv on the volume, so a foreign tool venv (an
   * MCP server another agent spawned) can map `tokenizers.pyd` through *its*
   * link and make *our* link refuse the exclusive open. Sharing is per file
   * but deletion is per link: unlinking our link while another link is mapped
   * succeeds (verified on Windows 11 with an exclusive open and with a
   * LoadLibrary mapping), so these block only when a process maps our own
   * link. Resolving that needs per-process module attribution (see
   * windows-restart-manager.ts), which is why they are reported separately.
   */
  shared: string[]
}

/**
 * A running image or a mapped module refuses an exclusive read/write open on
 * Windows with a sharing violation. ENOENT means nothing can hold the file.
 * The hard-link count is read only for files that refused the open, so the
 * common unlocked case costs one open per file.
 */
export function probeInstallResourceLocks(
  resources: readonly string[],
  options: { limit?: number; fsImpl?: LockProbeFs; platform?: NodeJS.Platform } = {}
): InstallResourceLocks {
  const platform = options.platform ?? process.platform
  const result: InstallResourceLocks = { definite: [], shared: [] }

  if (platform !== 'win32') {return result}

  const fsImpl = options.fsImpl ?? fs
  const limit = options.limit ?? Number.POSITIVE_INFINITY

  for (const target of resources) {
    let fd: number | undefined
    let locked = false

    try {
      fd = fsImpl.openSync(target, 'r+')
    } catch (error: any) {
      locked = Boolean(error) && error.code !== 'ENOENT'
    } finally {
      if (fd !== undefined) {
        try {
          fsImpl.closeSync(fd)
        } catch {
          void 0
        }
      }
    }

    if (!locked) {continue}

    let links = 1

    try {
      links = fsImpl.statSync(target).nlink
    } catch {
      links = 1
    }

    if (links > 1) {
      result.shared.push(target)
    } else {
      result.definite.push(target)
    }

    if (result.definite.length + result.shared.length >= limit) {break}
  }

  return result
}

/** Every file that refused an exclusive open, definite first. */
export function findLockedInstallResources(
  resources: readonly string[],
  options: { limit?: number; fsImpl?: LockProbeFs; platform?: NodeJS.Platform } = {}
): string[] {
  const locks = probeInstallResourceLocks(resources, options)

  return [...locks.definite, ...locks.shared]
}
