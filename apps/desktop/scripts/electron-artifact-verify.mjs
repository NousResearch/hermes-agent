/**
 * Official-release checksum verification for the Electron artifact zip.
 *
 * The prepared packaging path obtains Electron through electron-builder's
 * `downloadElectronArtifactZip`, which delegates to @electron/get. @electron/get verifies the
 * archive against the SHASUMS256.txt served by the SAME base URL it downloads from, and
 * ELECTRON_MIRROR (npmmirror, a corporate mirror, …) is honoured for both the archive and the
 * manifest. A mirror therefore authenticates its own content: a compromised mirror can serve a
 * substituted zip together with a matching checksums file and pass verification.
 *
 * This module anchors the archive to the OFFICIAL electron release manifest
 * (github.com/electron/electron) instead, so a mirror can mirror the bytes or fail — it cannot
 * substitute them. The official manifest is cached next to the artifact so repeat builds (and
 * offline builds after one online verification) do not re-fetch it.
 *
 * Two official channels are tried: the release download URL, then the GitHub releases API (for
 * networks that reach api.github.com but not github.com). GITHUB_TOKEN/GH_TOKEN, when present,
 * only ever goes to the API host — never to the release CDN.
 */

import { createHash } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

export const OFFICIAL_RELEASE_BASE = 'https://github.com/electron/electron/releases/download'
export const OFFICIAL_API_BASE = 'https://api.github.com'
export const REQUIRE_OFFICIAL_ENV = 'HERMES_ELECTRON_REQUIRE_OFFICIAL_CHECKSUM'

/**
 * Extract the SHA-256 a release manifest records for `filename`.
 * Lines are `<64 hex> *<filename>` (binary marker) or `<64 hex>  <filename>`.
 * @param {string} text
 * @param {string} filename
 * @returns {string | null}
 */
export function parseShasums256(text, filename) {
  if (typeof text !== 'string' || !filename) return null
  for (const line of text.split(/\r?\n/)) {
    const match = /^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$/.exec(line)
    if (match && match[2] === filename) return match[1].toLowerCase()
  }
  return null
}

/**
 * @param {string} file
 * @returns {Promise<string>}
 */
export async function sha256File(file) {
  const hash = createHash('sha256')
  for await (const chunk of fs.createReadStream(file)) hash.update(chunk)
  return hash.digest('hex')
}

/**
 * @param {string} url
 * @param {{ fetchImpl: typeof fetch, timeoutMs: number, headers?: Record<string, string> }} options
 * @returns {Promise<string>}
 */
async function requestText(url, { fetchImpl, timeoutMs, headers }) {
  const response = await fetchImpl(url, { signal: AbortSignal.timeout(timeoutMs), redirect: 'follow', headers })
  if (!response.ok) throw new Error(`HTTP ${response.status} fetching ${url}`)
  const text = await response.text()
  if (!text.trim()) throw new Error(`empty body at ${url}`)
  return text
}

/**
 * Fetch the official SHASUMS256.txt for a pinned Electron version.
 * @param {string} version
 * @param {{ fetchImpl?: typeof fetch, timeoutMs?: number, baseUrl?: string, apiBase?: string, token?: string | null }} [options]
 * @returns {Promise<{ text: string, url: string }>}
 */
export async function fetchOfficialShasums(version, options = {}) {
  const { fetchImpl = fetch, timeoutMs = 15000, baseUrl = OFFICIAL_RELEASE_BASE, apiBase = OFFICIAL_API_BASE } = options
  const token = options.token ?? process.env.GITHUB_TOKEN ?? process.env.GH_TOKEN ?? null
  const directUrl = `${baseUrl}/v${version}/SHASUMS256.txt`
  const failures = []
  try {
    return { text: await requestText(directUrl, { fetchImpl, timeoutMs }), url: directUrl }
  } catch (error) {
    failures.push(`${directUrl}: ${error instanceof Error ? error.message : error}`)
  }
  const releaseUrl = `${apiBase}/repos/electron/electron/releases/tags/v${version}`
  const headers = token ? { Authorization: `Bearer ${token}` } : {}
  try {
    const release = JSON.parse(await requestText(releaseUrl, { fetchImpl, timeoutMs, headers }))
    const asset = Array.isArray(release?.assets) ? release.assets.find(item => item?.name === 'SHASUMS256.txt') : null
    if (!asset?.url) throw new Error('release has no SHASUMS256.txt asset')
    const text = await requestText(asset.url, { fetchImpl, timeoutMs, headers: { ...headers, Accept: 'application/octet-stream' } })
    return { text, url: String(asset.url) }
  } catch (error) {
    failures.push(`${releaseUrl}: ${error instanceof Error ? error.message : error}`)
  }
  throw new Error(failures.join(' | '))
}

/** @param {string} cacheDir @param {string} version @returns {string} */
function cachedManifestPath(cacheDir, version) {
  return path.join(cacheDir, `SHASUMS256-v${version}.txt`)
}

/**
 * Verify `archivePath` against the official Electron release manifest.
 *
 * Throws when the archive does NOT match a hash the official manifest publishes for it — a
 * positive tamper signal — and deletes the poisoned archive so the next build re-downloads.
 * When the official manifest cannot be reached and no cached copy exists, returns
 * `{ verified: false }` after warning, unless strict mode (`requireOfficial` or
 * HERMES_ELECTRON_REQUIRE_OFFICIAL_CHECKSUM=1) turns that into a failure.
 *
 * @param {string} archivePath
 * @param {{ version: string, cacheDir?: string, fetchImpl?: typeof fetch, timeoutMs?: number, baseUrl?: string, apiBase?: string, token?: string | null, requireOfficial?: boolean, log?: Pick<Console, 'log' | 'warn'> }} options
 * @returns {Promise<{ verified: boolean, source: 'official' | 'cache' | null, sha256?: string, reason?: string }>}
 */
export async function verifyElectronArtifact(archivePath, options) {
  const {
    version, cacheDir, fetchImpl = fetch, timeoutMs = 15000, baseUrl = OFFICIAL_RELEASE_BASE,
    apiBase = OFFICIAL_API_BASE, log = console,
  } = options
  const requireOfficial = options.requireOfficial ?? process.env[REQUIRE_OFFICIAL_ENV] === '1'
  const filename = path.basename(archivePath)
  /** @type {{ text: string, source: 'official' | 'cache' } | null} */
  let manifest = null
  /** @type {string | null} */
  let unreachable = null
  try {
    const fetched = await fetchOfficialShasums(version, { fetchImpl, timeoutMs, baseUrl, apiBase, token: options.token })
    manifest = { text: fetched.text, source: 'official' }
    if (cacheDir) {
      try {
        fs.mkdirSync(cacheDir, { recursive: true })
        fs.writeFileSync(cachedManifestPath(cacheDir, version), fetched.text)
      } catch (error) {
        log.warn(`electron checksum: could not cache the official manifest: ${error instanceof Error ? error.message : error}`)
      }
    }
  } catch (error) {
    unreachable = error instanceof Error ? error.message : String(error)
    if (cacheDir) {
      try {
        manifest = { text: fs.readFileSync(cachedManifestPath(cacheDir, version), 'utf8'), source: 'cache' }
      } catch {
        // No cached manifest — the unreachable branch below decides warn vs fail-closed.
      }
    }
  }
  if (manifest === null) {
    const reason = `official Electron v${version} SHASUMS256.txt is unreachable (${unreachable}) and no cached copy exists`
    if (requireOfficial) throw new Error(`electron checksum: ${reason}`)
    log.warn(`electron checksum: ${reason}; ${filename} is UNVERIFIED against the official release`)
    return { verified: false, source: null, reason }
  }
  const expected = parseShasums256(manifest.text, filename)
  if (expected === null) {
    throw new Error(`electron checksum: the ${manifest.source} SHASUMS256.txt for v${version} does not list ${filename}`)
  }
  const actual = await sha256File(archivePath)
  if (actual !== expected) {
    fs.rmSync(archivePath, { force: true })
    throw new Error(`electron checksum: ${filename} does not match the official release hash (expected ${expected}, got ${actual}); the cached artifact was deleted — re-run the build to download a clean copy`)
  }
  log.log(`electron checksum: ${filename} verified against the ${manifest.source} SHASUMS256.txt (${actual.slice(0, 12)}…)`)
  return { verified: true, source: manifest.source, sha256: actual }
}
