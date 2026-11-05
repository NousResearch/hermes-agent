// remote-only-update-check.ts — version awareness for a light artifact.
//
// A light (remote-only) build has no Python payload and no native update
// owner: the App Installer checker runs inside the payload python, and the
// light feed subtree upstream does not publish. So the only forward-looking
// fact this build can carry is "a newer official release exists". This module
// reads it from GitHub's /releases/latest redirect (no API, so no 60 req/h
// limit), caches it in userData, and never downloads, stages or installs
// anything.

import fs from 'node:fs/promises'
import path from 'node:path'

export const REMOTE_ONLY_RELEASES_URL = 'https://github.com/NousResearch/hermes-agent/releases'
export const REMOTE_ONLY_LATEST_RELEASE_URL = `${REMOTE_ONLY_RELEASES_URL}/latest`
/** A release does not change intraday; a fresh cache keeps startup offline. */
export const REMOTE_ONLY_UPDATE_CACHE_TTL_MS = 12 * 60 * 60 * 1000

export interface RemoteOnlyUpdateStatus {
  /** False on every artifact that is not light: only remote-only builds use this check. */
  supported: boolean
  current: string
  latest: string | null
  newer: boolean
  releaseUrl: string
  checkedAt: number
  error?: string
}

interface CacheFile {
  checkedAt: number
  latest: string
}

/** `/releases/tag/v0.22.9` (absolute or relative) → `v0.22.9`. */
export function parseReleaseTag(location: string | null | undefined): string | null {
  if (!location) {
    return null
  }

  const match = /\/releases\/tag\/([^/?#]+)\/?(?:[?#].*)?$/.exec(location.trim())

  if (!match) {
    return null
  }

  try {
    return decodeURIComponent(match[1])
  } catch {
    return match[1]
  }
}

interface ParsedVersion {
  major: number
  minor: number
  patch: number
  prerelease: string[]
}

function parseVersion(value: string): ParsedVersion | null {
  const match = /^v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$/.exec(String(value ?? '').trim())

  if (!match) {
    return null
  }

  return {
    major: Number(match[1]),
    minor: Number(match[2]),
    patch: Number(match[3]),
    prerelease: match[4] ? match[4].split('.') : []
  }
}

function comparePrerelease(left: string[], right: string[]): number {
  // A release outranks any prerelease of the same version.
  if (left.length === 0 || right.length === 0) {
    return left.length === right.length ? 0 : left.length === 0 ? 1 : -1
  }

  for (let index = 0; index < Math.max(left.length, right.length); index += 1) {
    const a = left[index]
    const b = right[index]

    if (a === undefined) {
      return -1
    }

    if (b === undefined) {
      return 1
    }

    const aNumber = /^\d+$/.test(a)
    const bNumber = /^\d+$/.test(b)

    if (aNumber && bNumber) {
      const difference = Number(a) - Number(b)

      if (difference !== 0) {
        return difference < 0 ? -1 : 1
      }
    } else if (aNumber !== bNumber) {
      // Numeric identifiers have lower precedence than alphanumeric ones.
      return aNumber ? -1 : 1
    } else if (a !== b) {
      return a < b ? -1 : 1
    }
  }

  return 0
}

/** Semver precedence for release tags (`v0.22.9`), 0 when either side is not semver. */
export function compareVersions(left: string, right: string): number {
  const a = parseVersion(left)
  const b = parseVersion(right)

  if (!a || !b) {
    return 0
  }

  for (const key of ['major', 'minor', 'patch'] as const) {
    if (a[key] !== b[key]) {
      return a[key] < b[key] ? -1 : 1
    }
  }

  return comparePrerelease(a.prerelease, b.prerelease)
}

export function isNewerVersion(latest: string | null, current: string): boolean {
  return latest ? compareVersions(latest, current) > 0 : false
}

export interface RemoteOnlyUpdateCheckerDeps {
  currentVersion: string
  /** Where the last answer is cached; inside `app.getPath('userData')`. */
  cachePath: string
  ttlMs?: number
  now?: () => number
  fetch?: typeof fetch
  log?: (message: string) => void
}

export interface RemoteOnlyUpdateChecker {
  check(opts?: { force?: boolean }): Promise<RemoteOnlyUpdateStatus>
}

export function createRemoteOnlyUpdateChecker(deps: RemoteOnlyUpdateCheckerDeps): RemoteOnlyUpdateChecker {
  const ttlMs = deps.ttlMs ?? REMOTE_ONLY_UPDATE_CACHE_TTL_MS
  const now = deps.now ?? Date.now
  const doFetch = deps.fetch ?? fetch
  const log = deps.log ?? (() => {})

  async function readCache(): Promise<CacheFile | null> {
    try {
      const parsed = JSON.parse(await fs.readFile(deps.cachePath, 'utf8')) as Partial<CacheFile>

      if (typeof parsed?.latest === 'string' && parsed.latest && typeof parsed.checkedAt === 'number') {
        return { checkedAt: parsed.checkedAt, latest: parsed.latest }
      }
    } catch {
      // Missing, unreadable or corrupt: a fresh check is the answer.
    }

    return null
  }

  async function writeCache(latest: string, checkedAt: number): Promise<void> {
    try {
      await fs.mkdir(path.dirname(deps.cachePath), { recursive: true })
      await fs.writeFile(deps.cachePath, JSON.stringify({ checkedAt, latest }), 'utf8')
    } catch (error) {
      // A read-only userData must not turn a successful check into a failure.
      log(`[updates] remote-only cache write failed: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  function status(latest: string | null, checkedAt: number, error?: string): RemoteOnlyUpdateStatus {
    return {
      supported: true,
      current: deps.currentVersion,
      latest,
      newer: isNewerVersion(latest, deps.currentVersion),
      releaseUrl: latest ? `${REMOTE_ONLY_RELEASES_URL}/tag/${encodeURIComponent(latest)}` : REMOTE_ONLY_RELEASES_URL,
      checkedAt,
      ...(error ? { error } : {})
    }
  }

  return {
    async check({ force = false } = {}) {
      const cached = await readCache()

      if (!force && cached && now() - cached.checkedAt < ttlMs) {
        return status(cached.latest, cached.checkedAt)
      }

      try {
        const response = await doFetch(REMOTE_ONLY_LATEST_RELEASE_URL, {
          // Read the tag from the redirect itself: no API call, no token, no
          // rate limit, and nothing is downloaded.
          redirect: 'manual',
          headers: { accept: 'text/html' }
        })
        const latest = parseReleaseTag(response.headers.get('location'))

        if (!latest) {
          throw new Error(`GitHub did not redirect to a release tag (HTTP ${response.status})`)
        }

        const checkedAt = now()
        await writeCache(latest, checkedAt)
        log(`[updates] remote-only: latest release is ${latest} (this build is ${deps.currentVersion})`)

        return status(latest, checkedAt)
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        log(`[updates] remote-only version check failed: ${message}`)

        // A stale answer is more useful than "unknown": the user wants to know
        // which version to rebase onto.
        return cached ? status(cached.latest, cached.checkedAt, message) : status(null, now(), message)
      }
    }
  }
}
