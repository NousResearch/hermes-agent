// Makes every packaging binary download retried, resumable, and stall-aborted
// instead of dying on the first transport error (#122478). The vendored
// app-builder-lib fork retries only errors carrying .code; undici's uncoded
// TypeError: terminated and the fixed 10-minute AbortSignal budget that killed
// a 138 MB Electron zip at 50% on a slow link both propagate fatal, and every
// retry restarted from byte 0. This swaps @electron/get's system downloader in
// memory (registerHooks, node_modules is never mutated); a pinned-shape change
// fails loudly instead of silently losing the retries.
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createRequire, registerHooks } from 'node:module'
import { pathToFileURL } from 'node:url'

const require = createRequire(import.meta.url)
const sourceRoot = path.resolve(import.meta.dirname, '../../..')

/** A transfer may stall this long before the attempt is aborted; a healthy slow link is never aborted for being slow. */
export const STALL_TIMEOUT_MS = 90_000
export const MAX_ATTEMPTS = 6
/** Backoff between download attempts: 1s, 2s, 4s, capped at 8s. */
const RETRY_BACKOFF_MAX_MS = 8_000

/**
 * Transport-level failures worth another attempt. @electron/get v5 downloads via
 * fetch, whose connection resets surface as uncoded TypeError: terminated and
 * whose aborts as DOMException TimeoutError — neither carries .code, which is
 * exactly why the vendored fork's coded retry never saw them.
 * @param {unknown} error
 * @returns {boolean}
 */
export function isRetryableDownloadError(error) {
  if (error == null || typeof error !== 'object') return false
  const err = /** @type {Error & { code?: unknown }} */ (error)
  if (err.code === 'HERMES_RESUME_REFUSED') return true
  if (typeof err.code === 'string' && ['ENOTFOUND', 'ETIMEDOUT', 'ECONNRESET', 'EPIPE', 'ECONNREFUSED', 'EAI_AGAIN'].includes(err.code)) return true
  if (err.name === 'TimeoutError' || err.name === 'AbortError') return true
  if (typeof err.message !== 'string') return false
  if (err.name === 'TypeError' && /terminated|fetch failed|network/i.test(err.message)) return true
  if (/download stalled: no bytes/.test(err.message)) return true
  if (/\bcode\s+(429|5\d\d)\b/.test(err.message)) return true
  return false
}

/**
 * sha256 of a URL's directory part — the cache key @electron/get has used since
 * v2, so the shared cache the npm `electron` package populated during dependency
 * install can seed packaging with the byte-identical zip.
 * @param {string} url
 * @returns {string}
 */
export function electronCacheKey(url) {
  const parsed = new URL(url)
  parsed.hash = ''
  parsed.search = ''
  parsed.pathname = path.posix.dirname(parsed.pathname)
  return require('node:crypto').createHash('sha256').update(parsed.toString()).digest('hex')
}

/** @returns {string} shared cache root the npm electron package downloads into (envPaths('electron').cache semantics) */
export function sharedElectronCacheRoot() {
  if (process.platform === 'win32') {
    const localAppData = process.env.LOCALAPPDATA?.trim()
    if (localAppData) return path.join(localAppData, 'electron', 'Cache')
  } else if (process.platform === 'darwin') {
    return path.join(os.homedir(), 'Library', 'Caches', 'electron')
  }
  const xdg = process.env.XDG_CACHE_HOME?.trim()
  return xdg && path.isAbsolute(xdg) ? path.join(xdg, 'electron') : path.join(os.homedir(), '.cache', 'electron')
}

/**
 * Locate an artifact in the shared electron cache the npm `electron` package
 * already fetched. The caller lets @electron/get checksum-validate whatever we
 * hand it, so a stale shared copy can never silently ship: validation fails and
 * the install falls back to a real download on the next run.
 * @param {string} url @param {string} [sharedRoot] override for tests
 * @returns {string | null}
 */
export function findInSharedElectronCache(url, sharedRoot = sharedElectronCacheRoot()) {
  const fileName = path.basename(new URL(url).pathname)
  if (fileName === 'SHASUMS256.txt') return null
  const candidate = path.join(sharedRoot, electronCacheKey(url), fileName)
  try {
    const stat = fs.statSync(candidate)
    if (stat.isFile() && stat.size > 0) return candidate
  } catch { /* miss */ }
  return null
}

/** @param {unknown} error @returns {string} */
function describeError(error) {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error)
}

/**
 * One transfer attempt: fetch with a stall watchdog instead of a fixed minute
 * budget, appending to the partial file via HTTP Range when resuming. A caller
 * `signal` is deliberately superseded — the fixed 10-minute AbortSignal is what
 * aborted healthy slow transfers in the first place.
 * @param {string} url @param {string} targetFilePath
 * @param {Record<string, unknown>} [options] fetch options plus getProgressCallback
 * @param {number} [startOffset] bytes already on disk from a previous attempt
 * @param {number} [stallTimeoutMs] test hook for {@link STALL_TIMEOUT_MS}
 * @returns {Promise<void>}
 */
export async function resumableFetchToFile(url, targetFilePath, options = {}, startOffset = 0, stallTimeoutMs = STALL_TIMEOUT_MS) {
  const { getProgressCallback, ...fetchOptions } = options
  const controller = new AbortController()
  let lastChunkAt = Date.now()
  const watchdog = setInterval(() => {
    if (Date.now() - lastChunkAt > stallTimeoutMs) controller.abort(new Error(`download stalled: no bytes for ${Math.round(stallTimeoutMs / 1000)}s`))
  }, Math.max(250, Math.round(stallTimeoutMs / 10)))
  try {
    const headers = startOffset > 0 ? { range: `bytes=${startOffset}-` } : undefined
    const response = await fetch(url, { ...fetchOptions, headers, signal: controller.signal })
    if (!response.ok) {
      const body = await response.text().catch(() => '')
      throw new Error(`Response code ${response.status} (${response.statusText}) for ${url}${body ? ` — ${body.slice(0, 200)}` : ''}`)
    }
    if (startOffset > 0 && response.status !== 206) {
      // Server ignored the range: appending would corrupt the artifact.
      throw Object.assign(new Error('resume refused: server returned 200 for a ranged request'), { code: 'HERMES_RESUME_REFUSED' })
    }
    if (!response.body) throw new Error('Response body is empty')
    const { pipeline } = await import('node:stream/promises')
    const { Readable } = await import('node:stream')
    const contentLength = parseInt(response.headers.get('content-length') || '', 10)
    const total = Number.isFinite(contentLength) && contentLength >= 0 ? startOffset + contentLength : null
    let transferred = startOffset
    const report = () => {
      if (getProgressCallback) void /** @type {() => unknown} */ (getProgressCallback)({ transferred, total, percent: total ? transferred / total : 0 })
    }
    report()
    const write = startOffset > 0 ? fs.createWriteStream(targetFilePath, { flags: 'a' }) : fs.createWriteStream(targetFilePath)
    await pipeline(Readable.fromWeb(/** @type {import('node:stream/web').ReadableStream} */ (response.body)), async function* (source) {
      for await (const chunk of source) {
        lastChunkAt = Date.now()
        transferred += chunk.length
        yield chunk
      }
    }, write)
    report()
  } finally {
    clearInterval(watchdog)
    controller.abort() // release the response body
  }
}

/**
 * Download with bounded retries, backoff, and HTTP Range resume of the partial
 * file the previous attempt left behind.
 * @param {string} url @param {string} targetFilePath @param {Record<string, unknown>} [options]
 */
async function downloadWithRetryResume(url, targetFilePath, options = {}) {
  fs.mkdirSync(path.dirname(targetFilePath), { recursive: true })
  for (let attempt = 1; ; attempt++) {
    const startOffset = fs.existsSync(targetFilePath) ? fs.statSync(targetFilePath).size : 0
    try {
      await resumableFetchToFile(url, targetFilePath, options, startOffset)
      return
    } catch (error) {
      if ((/** @type {Error & { code?: unknown }} */ (error)).code === 'HERMES_RESUME_REFUSED' && attempt < MAX_ATTEMPTS) {
        fs.rmSync(targetFilePath, { force: true })
        continue
      }
      if (attempt >= MAX_ATTEMPTS || !isRetryableDownloadError(error)) throw error
      const delay = Math.min(2 ** (attempt - 1) * 1_000, RETRY_BACKOFF_MAX_MS)
      console.warn(`      download attempt ${attempt}/${MAX_ATTEMPTS} failed (${describeError(error)}); retrying in ${delay / 1000}s${startOffset > 0 ? `, resuming from byte ${startOffset}` : ''}`)
      await new Promise(resolve => setTimeout(resolve, delay))
    }
  }
}

/**
 * Replacement system downloader for @electron/get. downloadArtifact still owns
 * cache lookup, checksum validation, and cache writes; on its miss path it hands
 * us the temp file it owns, so seeding from the shared electron cache (the npm
 * `electron` package already fetched the byte-identical zip) and resumable
 * fetching slot in underneath all of that.
 */
export class ResumableDownloader {
  /** @param {{ sharedRoot?: string }} [options] */
  constructor({ sharedRoot } = {}) {
    this.sharedRoot = sharedRoot
  }

  /** @param {string} url @param {string} targetFilePath @param {Record<string, unknown>} [options] */
  async download(url, targetFilePath, options = {}) {
    const seeded = findInSharedElectronCache(url, this.sharedRoot)
    if (seeded) {
      console.log(`      reusing electron artifact already on disk: ${path.basename(seeded)}`)
      await fs.promises.copyFile(seeded, targetFilePath)
      return
    }
    await downloadWithRetryResume(url, targetFilePath, options)
  }
}

/** The exact resolver body this patch replaces; must match @electron/get 5.1.0's compiled output. */
const RESOLVER_NEEDLE = 'export async function getDownloaderForSystem() {\n    return new FetchDownloader();\n}'

/**
 * Pure transform of @electron/get's downloader-resolver source. Exported for
 * tests; an upstream shape change throws rather than silently losing retries.
 * @param {string} source @param {string} selfUrl
 * @returns {string}
 */
export function patchResolverSource(source, selfUrl) {
  if (!source.includes(RESOLVER_NEEDLE)) {
    throw new Error('@electron/get downloader-resolver shape changed; revalidate patch-electron-get-downloads.mjs')
  }
  return source.replace(RESOLVER_NEEDLE, `export async function getDownloaderForSystem() {
    const { ResumableDownloader } = await import(${JSON.stringify(selfUrl)});
    return new ResumableDownloader();
}`)
}

let installed = false

/**
 * Swap @electron/get's system downloader for {@link ResumableDownloader} via an
 * in-memory module hook. Idempotent. Missing node dependencies are tolerated
 * (nothing to patch; the run fails later with its own clear error), but a
 * version or shape mismatch fails closed like the other vendored patches.
 * @param {{ require?: NodeJS.Require, register?: typeof registerHooks }} [injectable]
 * @returns {string | null} the patched resolver module URL, or null when node dependencies are absent
 */
export function installElectronGetDownloadPatch({ require: doRequire = require, register = registerHooks } = {}) {
  if (installed) return lastInstallResult
  let entry
  try {
    // Resolve from the source root so the builder's root-level @electron/get
    // (5.1.0) is found, not the desktop workspace copy the npm electron package pins.
    const rootRequire = createRequire(doRequire.resolve(path.join(sourceRoot, 'package.json')))
    entry = rootRequire.resolve('@electron/get')
  } catch {
    lastInstallResult = null
    return null // node dependencies not installed yet; nothing to patch
  }
  const packageRoot = path.dirname(path.dirname(entry))
  const version = JSON.parse(fs.readFileSync(path.join(packageRoot, 'package.json'), 'utf8')).version
  if (version !== '5.1.0') {
    throw new Error(`Installed @electron/get ${version} is not the pinned 5.1.0; revalidate patch-electron-get-downloads.mjs`)
  }
  const resolverUrl = pathToFileURL(path.join(packageRoot, 'dist', 'downloader-resolver.js')).href
  register({
    load(url, context, nextLoad) {
      const loaded = nextLoad(url, context)
      if (url !== resolverUrl) return loaded
      return { ...loaded, source: patchResolverSource(loaded.source.toString(), import.meta.url) }
    },
  })
  installed = true
  lastInstallResult = resolverUrl
  return resolverUrl
}

/** @type {string | null | undefined} */
let lastInstallResult

// Self-install on import: every consumer that is about to load app-builder-lib
// (prepare-packaging-tools, windows-bundle-tools, preload usage) gets the patched
// downloader registered before its first dynamic import resolves @electron/get.
// Missing node dependencies are tolerated; a pinned-shape mismatch throws.
installElectronGetDownloadPatch()
