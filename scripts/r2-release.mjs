#!/usr/bin/env node
// scripts/r2-release.mjs — zero-dependency Cloudflare R2 (S3 API) client for
// release artifacts. Runs on every release runner (node is guaranteed there
// via pm/lock.json; the notes + prune jobs need no npm ci), so the only
// dependency is node's own fetch + crypto. SigV4 signed exactly per the
// botocore-generated vectors in tests-js/r2-release.test.mjs.
//
//   node scripts/r2-release.mjs put --tag vX.Y.Z --key <filename> --file <path>
//   node scripts/r2-release.mjs finalize --tag vX.Y.Z --dir <staging-dir>
//   node scripts/r2-release.mjs list [--prefix <p>]
//   node scripts/r2-release.mjs prune-canaries --keep-days 14 [--dry-run]
//
// Env (all required except where noted):
//   CLOUDFLARE_R2_ACCOUNT_ID       → S3 endpoint https://<account>.r2.cloudflarestorage.com
//   CLOUDFLARE_R2_ACCESS_KEY_ID    R2 API token (S3-compatible)
//   CLOUDFLARE_R2_SECRET_ACCESS_KEY
//   CLOUDFLARE_R2_BUCKET
//
// Bucket layout (matches app-updater.ts's D1-settled arms):
//   releases/tag/<tag>/<filename>          immutable per-release staging/archive
//   releases/win32/<channel>/<channel>.appinstaller   App Installer feed
//     releases/win32/<channel>/*.msixbundle           (produced by the
//                                                     publish-win32-updater job)
//   releases/darwin/<channel>/<channel>-mac.yml   electron-updater feed
//     releases/darwin/<channel>/*.{dmg,zip,blockmap}
// where <channel> is stable | canary (from the tag: -canary. → canary).
// The publish-win32-updater job merges the win32 legs' staging into the
// win32 feed; the darwin feed merge (r2 finalize) is currently DISABLED
// (no macOS updater arm).

import { createHash, createHmac } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { contentTypeFor } from './msix-shared.mjs'

// Re-exported for the r2 test (the Content-Type mapping is shared with the
// stage job; msix-shared.mjs is the single source).
export { contentTypeFor } from './msix-shared.mjs'

const REGION = 'auto'
const SERVICE = 's3'
const EMPTY_SHA = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'

// ---------------------------------------------------------------------------
// SigV4 (pure; the test file pins these against botocore-generated vectors)
// ---------------------------------------------------------------------------

/** RFC3986 encode: everything except unreserved [A-Za-z0-9-_.~]. */
export function rfc3986Encode(value) {
  return encodeURIComponent(value).replace(/[!'()*]/g, (c) =>
    '%' + c.charCodeAt(0).toString(16).toUpperCase(),
  )
}

/** Canonical query string: params sorted by encoded key (then encoded value). */
export function canonicalQuery(params) {
  return Object.entries(params)
    .map(([k, v]) => [rfc3986Encode(k), rfc3986Encode(String(v))])
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([k, v]) => `${k}=${v}`)
    .join('&')
}

/**
 * Canonical request for one S3-style request.
 * `headers` is the exact header set that will be sent (host, x-amz-date,
 * x-amz-content-sha256); canonicalization lowercases + sorts them.
 */
export function canonicalRequest(method, path, query, headers, payloadHash) {
  const names = Object.keys(headers).map((n) => n.toLowerCase()).sort()
  // Header values are read case-insensitively: keys may be mixed-case
  // ('Content-Type'), but SigV4 canonicalizes the NAME to lowercase, so
  // `headers[lowerName]` would miss the value. Find the original-key match.
  const valueFor = (name) => {
    const key = Object.keys(headers).find((k) => k.toLowerCase() === name)
    return key == null ? undefined : headers[key]
  }
  const canonicalHeaders = names
    .map((n) => `${n}:${String(valueFor(n)).trim().replace(/\s+/g, ' ')}`)
    .join('\n')
  return [
    method,
    path,
    query,
    canonicalHeaders,
    '',
    names.join(';'),
    payloadHash,
  ].join('\n')
}

/** The AWS4 string-to-sign for a canonical request. */
export function stringToSign(canonical, date, scope) {
  return ['AWS4-HMAC-SHA256', date, scope, createHash('sha256').update(canonical).digest('hex')].join('\n')
}

function hmac(key, msg) {
  return createHmac('sha256', key).update(msg).digest()
}

/** Signature for a string-to-sign, given secret key + credential scope parts. */
export function signature(stringToSignText, secretKey, date, region, service) {
  const kDate = hmac(`AWS4${secretKey}`, date)
  const kRegion = hmac(kDate, region)
  const kService = hmac(kRegion, service)
  const kSigning = hmac(kService, 'aws4_request')
  return hmac(kSigning, stringToSignText).toString('hex')
}

/**
 * Full AWS4-HMAC-SHA256 Authorization header value for one request.
 * `now` is 'YYYYMMDDTHHMMSSZ' (injectable for tests); scope date is its
 * first 8 chars. `payloadHash` is hex sha256 of the body (or the empty
 * string hash for bodyless requests).
 */
export function authHeader({ method, host, path, query, headers, payloadHash, accessKeyId, secretKey, now, region = REGION, service = SERVICE }) {
  const date = now.slice(0, 8)
  const canonical = canonicalRequest(method, path, query, headers, payloadHash)
  const sts = stringToSign(canonical, now, `${date}/${region}/${service}/aws4_request`)
  const sig = signature(sts, secretKey, date, region, service)
  const signedHeaders = Object.keys(headers).map((n) => n.toLowerCase()).sort().join(';')
  return (
    `AWS4-HMAC-SHA256 Credential=${accessKeyId}/${date}/${region}/${service}/aws4_request, ` +
    `SignedHeaders=${signedHeaders}, Signature=${sig}`
  )
}

// ---------------------------------------------------------------------------
// R2 request plumbing
// ---------------------------------------------------------------------------

export function s3Endpoint(accountId) {
  return `https://${accountId}.r2.cloudflarestorage.com`
}

/** Encode an object key into the URI path, segment by segment. */
export function encodeKeyPath(key) {
  return key.split('/').map(rfc3986Encode).join('/')
}

function requiredEnv(name) {
  const value = process.env[name]
  if (!value) {
    console.error(`::error::missing env ${name} — see the header comment in scripts/r2-release.mjs`)
    process.exit(2)
  }
  return value
}

function r2Headers(method, host, path, query, bodyHash, now, creds, contentType, contentLength) {
  const headers = {
    host,
    'x-amz-date': now,
    'x-amz-content-sha256': bodyHash,
  }
  // Optional Content-Type for binary artifacts (App Installer / MSIX). It is
  // added BEFORE the Authorization header is computed, so it lands in the
  // SigV4 canonical headers + SignedHeaders exactly like host / x-amz-*.
  if (contentType) headers['Content-Type'] = contentType
  // Stream bodies (files >2GiB) need an explicit Content-Length — R2
  // rejects a bodyless-length PUT with 411 MissingContentLength.
  if (contentLength != null) headers['Content-Length'] = String(contentLength)
  headers.authorization = authHeader({
    method,
    host,
    path,
    query,
    headers,
    payloadHash: bodyHash,
    accessKeyId: creds.accessKeyId,
    secretKey: creds.secretKey,
    now,
  })
  return headers
}

async function signedFetch(method, url, { body, bodyHash, creds, now, contentType, contentLength }) {
  const { host, pathname, search } = new URL(url)
  const query = search.replace(/^\?/, '')
  const headers = r2Headers(method, host, pathname, query, bodyHash, now, creds, contentType, contentLength)
  const res = await fetch(url, {
    method,
    headers,
    body: body ?? undefined,
    // Stream bodies (files >2GiB) require the duplex option on Node's fetch.
    ...(body != null && typeof body.pipe === 'function' ? { duplex: 'half' } : {}),
  })
  const text = await res.text()
  if (!res.ok) {
    console.error(`::error::R2 ${method} ${pathname} -> ${res.status}`)
    if (text) console.error(text.slice(0, 2000))
  }
  return { res, text }
}

async function retry(fn, tries = 3) {
  let lastError
  for (let attempt = 1; attempt <= tries; attempt++) {
    try {
      return await fn()
    } catch (err) {
      lastError = err
      if (attempt < tries) await new Promise((r) => setTimeout(r, 1000 * attempt))
    }
  }
  throw lastError
}

// ---------------------------------------------------------------------------
// Layout helpers (pure)
// ---------------------------------------------------------------------------

/** 'stable' for a stable tag, 'canary' for a -canary.<ts> tag. */
export function channelForTag(tag) {
  return /-canary\.20\d{6}(?:\d{6})?$/.test(tag) ? 'canary' : 'stable'
}

// Content-Type for MSIX / App Installer artifacts lives in msix-shared.mjs
// (single source — the same suffixes must drive the stage job's uploads).

/** Immutable per-release staging key. */
export function stagingKeyFor(tag, filename) {
  return `releases/tag/${tag}/${filename}`
}

/** The feed directory key for a platform arm + channel, e.g. releases/win32/stable/. */
export function feedDirFor(platform, channel) {
  return `releases/${platform}/${channel}`
}

/**
 * Merge per-leg electron-updater feed ymls (same channel + platform) into one.
 * Each leg's yml (mac: latest-mac.yml / canary-mac.yml) lists only its own
 * arch's files[]; the merged yml keeps the top-level fields of the first leg
 * (version/releaseDate) and the trailing top-level path/sha512, with the
 * files[] entries concatenated and deduped by url. Idempotent: merging an
 * already-merged yml is a no-op.
 *
 * Structure of a feed yml (electron-builder):
 *   version: ...
 *   files:
 *     - url: ...        ← entries (indented list items)
 *       sha512: ...
 *       size: ...
 *   path: ...           ← trailing top-level fields
 *   sha512: ...
 *   releaseDate: ...
 */
export function mergeFeedYmls(ymls) {
  if (ymls.length === 0) return ''
  const seen = new Set()
  const entries = []

  for (const yml of ymls) {
    // Split at the files: marker. Entries are the indented list items
    // (lines starting with whitespace + '- url:'); the tail is everything
    // after the last entry (top-level path/sha512/releaseDate).
    const filesIdx = yml.indexOf('\nfiles:')
    if (filesIdx === -1) continue
    const body = yml.slice(filesIdx + 1) // starts right after '\nfiles:'
    const lines = body.split('\n')
    let i = 0
    // Skip the 'files:' line itself.
    if (lines[0].trim() === '') i = 1
    // Collect entry blocks: lines starting with '- url:' plus their
    // following indented sha512/size lines.
    while (i < lines.length) {
      const line = lines[i]
      if (/^\s*-\s+url:/.test(line)) {
        const block = [line]
        let j = i + 1
        while (j < lines.length && /^\s+(?:sha512|size):/.test(lines[j])) {
          block.push(lines[j])
          j++
        }
        const urlMatch = line.match(/url:\s*([^\s]+)/)
        if (urlMatch && !seen.has(urlMatch[1])) {
          seen.add(urlMatch[1])
          entries.push(block.join('\n'))
        }
        i = j
      } else {
        i++
      }
    }
  }

  const first = ymls[0]
  const firstFilesIdx = first.indexOf('\nfiles:')
  const head = firstFilesIdx === -1 ? first : first.slice(0, firstFilesIdx)

  // Tail: everything after the LAST entry block in the FIRST yml (the
  // top-level path/sha512/releaseDate lines).
  let tail = ''
  {
    const body = first.slice(firstFilesIdx + 1)
    const lines = body.split('\n')
    let lastEntryEnd = -1
    for (let i = 0; i < lines.length; i++) {
      if (/^\s*-\s+url:/.test(lines[i])) {
        let j = i + 1
        while (j < lines.length && /^\s+(?:sha512|size):/.test(lines[j])) j++
        lastEntryEnd = j
        i = j - 1
      }
    }
    if (lastEntryEnd !== -1) {
      tail = lines.slice(lastEntryEnd).join('\n').replace(/^\n+/, '')
    }
  }

  const body = ['files:', ...entries].join('\n')
  const parts = [head, body]
  if (tail.trim() !== '') parts.push(tail)
  return parts.join('\n').replace(/\n{3,}/g, '\n\n').trimEnd() + '\n'
}

/**
 * Rewrite a feed yml's path:/url: entries to ABSOLUTE /releases/tag/<tag>/
 * object keys. The feed manifest lives in releases/darwin/<channel>/ but the
 * binaries live once in the tag archive; both updater mechanisms resolve the
 * value against the feed host root (electron-updater: new URL(path, baseUrl)).
 * `absKey` maps a filename to its absolute key (e.g. /releases/tag/v0.28.0/x).
 * Values that already start with '/' are left alone (idempotent).
 */
export function rewriteFeedPaths(ymlText, absKey) {
  return ymlText.replace(/^(\s*(?:-\s+)?(?:path|url)):\s*([^\s#]+)\s*$/gm, (_m, key, value) => {
    if (value.startsWith('/')) return _m
    return `${key}: ${absKey(value)}`
  })
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

async function putObject(creds, base, bucket, key, payload, now, contentType) {
  // `payload` is either a small in-memory Buffer (feed manifests from
  // finalize) or a FILE PATH (binaries via `put`). The msixbundle is
  // ~2.7GB so path payloads stream from disk (fs.readFileSync throws
  // ERR_FS_FILE_TOO_LARGE past 2GiB); buffers upload directly.
  const isPath = typeof payload === 'string'
  const size = isPath ? (await fs.promises.stat(payload)).size : payload.length
  const bodyHash = isPath
    ? await new Promise((resolve, reject) => {
        const hash = createHash('sha256')
        const stream = fs.createReadStream(payload)
        stream.on('data', (c) => hash.update(c))
        stream.on('end', () => resolve(hash.digest('hex')))
        stream.on('error', reject)
      })
    : createHash('sha256').update(payload).digest('hex')
  const url = `${base}/${bucket}/${encodeKeyPath(key)}`
  await retry(async () => {
    // Path payloads get a fresh stream per attempt: a consumed
    // ReadableStream cannot be replayed for the retry ("body object
    // should not be disturbed").
    const body = isPath ? fs.createReadStream(payload) : payload
    const { res, text } = await signedFetch('PUT', url, { body, bodyHash, contentLength: size, creds, now, contentType })
    if (!res.ok) throw new Error(`PUT ${key} -> ${res.status}${text ? `: ${text.slice(0, 300)}` : ''}`)
  })
  const { res: headRes, text: headText } = await retry(async () => {
    const r = await signedFetch('HEAD', url, { bodyHash: EMPTY_SHA, creds, now })
    if (!r.res.ok) throw new Error(`HEAD ${key} -> ${r.res.status}`)
    return r
  })
  const remoteSize = headRes.headers.get('content-length')
  if (remoteSize === null || String(remoteSize) !== String(size)) {
    console.error(`::error::R2 HEAD ${key}: size mismatch (remote ${remoteSize}, local ${size})`)
    process.exit(1)
  }
  console.log(`✓ r2: ${key} (${size} bytes)`)
}

async function cmdPut({ tag, key, file, keyIsFull = false }) {
  const accountId = requiredEnv('CLOUDFLARE_R2_ACCOUNT_ID')
  const accessKeyId = requiredEnv('CLOUDFLARE_R2_ACCESS_KEY_ID')
  const secretKey = requiredEnv('CLOUDFLARE_R2_SECRET_ACCESS_KEY')
  const bucket = requiredEnv('CLOUDFLARE_R2_BUCKET')
  const creds = { accessKeyId, secretKey }
  const base = s3Endpoint(accountId)

  const now = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}/, '')
  const keyPath = keyIsFull ? key : stagingKeyFor(tag, key)
  await putObject(creds, base, bucket, keyPath, file, now, contentTypeFor(key))
}

/**
 * finalize: write the per-channel feed MANIFESTS that point at the staged
 * binaries. Binaries live ONCE under releases/tag/<tag>/ (staged by the
 * matrix legs); the feed dirs carry only the manifests:
 *
 *   releases/darwin/<channel>/<channel>-mac.yml
 *     merged electron-updater feed; path:/url: entries rewritten to the
 *     absolute /releases/tag/<tag>/<file> locations (electron-updater
 *     resolves them with new URL(path, baseUrl)).
 *
 * The win32 App Installer feed (.appinstaller + .msixbundle per channel
 * dir) is produced by the msixbundle job (scripts/stage-msixbundle.mjs),
 * which uploads the manifests directly — nothing for finalize to merge.
 *
 * Expects --dir to contain the merged METADATA for ONE tag (the build
 * matrix uploads only this — the binaries go straight to R2 from each
 * leg and are never round-tripped through artifacts):
 *   *-mac.yml              the per-leg electron-updater feed files
 */
async function cmdFinalize({ tag, dir }) {
  const accountId = requiredEnv('CLOUDFLARE_R2_ACCOUNT_ID')
  const accessKeyId = requiredEnv('CLOUDFLARE_R2_ACCESS_KEY_ID')
  const secretKey = requiredEnv('CLOUDFLARE_R2_SECRET_ACCESS_KEY')
  const bucket = requiredEnv('CLOUDFLARE_R2_BUCKET')
  const creds = { accessKeyId, secretKey }
  const base = s3Endpoint(accountId)
  const channel = channelForTag(tag)
  const now = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}/, '')

  const files = fs.readdirSync(dir).filter((f) => fs.statSync(path.join(dir, f)).isFile())
  // Absolute key of a staged artifact — the feed manifests reference these.
  const absKey = (filename) => `/${stagingKeyFor(tag, filename)}`

  // --- darwin: merged feed yml only (dmg/zip/blockmap stay in the tag dir) ---
  const macYmls = files.filter((f) => f.endsWith(`-mac.yml`))
  if (macYmls.length > 0) {
    const darDir = feedDirFor('darwin', channel)
    const macFeedName = `${channel}-mac.yml`
    // Rewrite path:/url: to absolute /releases/tag/<tag>/ locations so the
    // client fetches binaries from the archive, not the feed dir.
    const merged = rewriteFeedPaths(mergeFeedYmls(macYmls.map((f) => fs.readFileSync(path.join(dir, f), 'utf8'))), absKey)
    await putObject(creds, base, bucket, `${darDir}/${macFeedName}`, Buffer.from(merged, 'utf8'), now)
  }

  console.log(`✓ r2: finalized ${tag} → ${channel} feed manifests`)
}

/** Parse a ListObjectsV2 XML body into { keys: string[], truncated, nextToken }. */
export function parseListXml(xml) {
  const unescape = (s) =>
    s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&amp;/g, '&')
  const keys = [...xml.matchAll(/<Key>([^<]+)<\/Key>/g)].map((m) => unescape(m[1]))
  const truncated = /<IsTruncated>true<\/IsTruncated>/.test(xml)
  const tokenMatch = xml.match(/<NextContinuationToken>([^<]+)<\/NextContinuationToken>/)
  return { keys, truncated, nextToken: tokenMatch ? unescape(tokenMatch[1]) : null }
}

async function listObjects(prefix = '') {
  const accountId = requiredEnv('CLOUDFLARE_R2_ACCOUNT_ID')
  const accessKeyId = requiredEnv('CLOUDFLARE_R2_ACCESS_KEY_ID')
  const secretKey = requiredEnv('CLOUDFLARE_R2_SECRET_ACCESS_KEY')
  const bucket = requiredEnv('CLOUDFLARE_R2_BUCKET')
  const creds = { accessKeyId, secretKey }
  const base = s3Endpoint(accountId)

  const keys = []
  let token = null
  for (;;) {
    const params = { 'list-type': '2', 'max-keys': '1000' }
    if (prefix) params.prefix = prefix
    if (token) params['continuation-token'] = token
    const query = canonicalQuery(params)
    const now = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}/, '')
    const url = `${base}/${bucket}?${query}`
    const { res, text } = await signedFetch('GET', url, { bodyHash: EMPTY_SHA, creds, now })
    if (!res.ok) process.exit(1)
    const parsed = parseListXml(text)
    keys.push(...parsed.keys)
    if (!parsed.truncated || !parsed.nextToken) break
    token = parsed.nextToken
  }
  return keys
}

async function cmdList({ prefix }) {
  for (const key of await listObjects(prefix)) console.log(key)
}

/** Keys whose own canary date (YYYYMMDD in the name) is before `cutoff`. */
export function canaryDoomedKeys(keys, cutoff) {
  return keys.filter((key) => {
    const m = key.match(/-canary\.(\d{8})/)
    return m && m[1] < cutoff
  })
}

async function cmdPrune({ keepDays, dryRun }) {
  const accountId = requiredEnv('CLOUDFLARE_R2_ACCOUNT_ID')
  const accessKeyId = requiredEnv('CLOUDFLARE_R2_ACCESS_KEY_ID')
  const secretKey = requiredEnv('CLOUDFLARE_R2_SECRET_ACCESS_KEY')
  const bucket = requiredEnv('CLOUDFLARE_R2_BUCKET')
  const creds = { accessKeyId, secretKey }
  const base = s3Endpoint(accountId)

  // Cutoff dated by the canary suffix in the KEY (like release.py's
  // --prune-canaries: a re-uploaded old tag never resets its clock).
  const cutoff = new Date(Date.now() - keepDays * 86400_000).toISOString().slice(0, 10).replace(/-/g, '')
  const keys = await listObjects()
  const doomed = canaryDoomedKeys(keys, cutoff)
  if (doomed.length === 0) {
    console.log(`✓ r2: no canary objects older than ${keepDays} days`)
    return
  }
  const now = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}/, '')
  for (const key of doomed.sort()) {
    if (dryRun) {
      console.log(`(dry-run) would delete r2:${key}`)
      continue
    }
    const url = `${base}/${bucket}/${encodeKeyPath(key)}`
    const { res } = await signedFetch('DELETE', url, { bodyHash: EMPTY_SHA, creds, now })
    if (!res.ok) process.exit(1)
    console.log(`deleted r2:${key}`)
  }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

function usage() {
  console.error(`usage:
  node scripts/r2-release.mjs put --tag vX.Y.Z --key <filename> --file <path> [--key-is-full]
  node scripts/r2-release.mjs finalize --tag vX.Y.Z --dir <staging-dir>
  node scripts/r2-release.mjs list [--prefix <p>]
  node scripts/r2-release.mjs prune-canaries --keep-days <n> [--dry-run]

  --key-is-full: the --key is a FULL object key (e.g. releases/win32/stable/…),
                 not a filename to archive under releases/tag/<tag>/.`)
  process.exit(2)
}

export function isMain() {
  return process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]
}

export async function main(argv = process.argv.slice(2)) {
  const [cmd, ...rest] = argv
  const args = {}
  for (let i = 0; i < rest.length; i++) {
    const flag = rest[i]
    if (['--tag', '--key', '--file', '--prefix', '--keep-days', '--dir'].includes(flag)) {
      args[flag.slice(2)] = rest[++i]
    } else if (flag === '--dry-run' || flag === '--key-is-full') {
      args[flag.slice(2)] = true
    } else {
      usage()
    }
  }
  if (cmd === 'put') {
    const tag = args.tag || process.env.HERMES_PAYLOAD_TAG
    if (!tag || !args.key || !args.file) usage()
    await cmdPut({ tag, key: args.key, file: args.file, keyIsFull: Boolean(args['key-is-full']) })
  } else if (cmd === 'finalize') {
    if (!args.tag || !args.dir) usage()
    await cmdFinalize({ tag: args.tag, dir: args.dir })
  } else if (cmd === 'list') {
    await cmdList({ prefix: args.prefix ?? '' })
  } else if (cmd === 'prune-canaries') {
    const keepDays = Number(args['keep-days'])
    if (!Number.isFinite(keepDays) || keepDays <= 0) usage()
    await cmdPrune({ keepDays, dryRun: Boolean(args.dryRun) })
  } else {
    usage()
  }
}

if (isMain()) {
  main().catch((err) => {
    console.error(`::error::${err?.stack ?? err}`)
    process.exit(1)
  })
}
