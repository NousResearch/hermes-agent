import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { afterEach, test } from 'vitest'
import {
  OFFICIAL_API_BASE, OFFICIAL_RELEASE_BASE, fetchOfficialShasums, parseShasums256, sha256File, verifyElectronArtifact,
} from './electron-artifact-verify.mjs'

/** @type {string[]} */
const roots = []
afterEach(() => roots.splice(0).forEach(root => fs.rmSync(root, { recursive: true, force: true })))

const quiet = { log() {}, warn() {} }
const VERSION = '40.1.0'
const DIRECT = `${OFFICIAL_RELEASE_BASE}/v${VERSION}/SHASUMS256.txt`
const RELEASE_API = `${OFFICIAL_API_BASE}/repos/electron/electron/releases/tags/v${VERSION}`

/** @param {Buffer | string} content @returns {{ root: string, archive: string, filename: string, sha: string }} */
function fixture(content) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'electron-verify-'))
  roots.push(root)
  const filename = 'electron-v40.1.0-darwin-arm64.zip'
  const archive = path.join(root, filename)
  const bytes = Buffer.isBuffer(content) ? content : Buffer.from(content)
  fs.writeFileSync(archive, bytes)
  return { root, archive, filename, sha: createHash('sha256').update(bytes).digest('hex') }
}

/** @param {string} filename @param {string} sha */
function manifest(filename, sha) {
  return `# electron release shasums\n${'0'.repeat(64)} *electron-v40.1.0-win32-x64.zip\n${sha} *${filename}\n`
}

/**
 * @param {Record<string, string | Error | { status: number } | { json: unknown }>} routes
 * @returns {{ fetchImpl: typeof fetch, calls: { url: string, headers: Record<string, string> }[] }}
 */
function fakeFetch(routes) {
  /** @type {{ url: string, headers: Record<string, string> }[]} */
  const calls = []
  /** @type {typeof fetch} */
  const fetchImpl = async (input, init = {}) => {
    const url = String(input)
    calls.push({ url, headers: /** @type {Record<string, string>} */ (init.headers ?? {}) })
    if (!(url in routes)) throw new Error(`unexpected fetch: ${url}`)
    const route = routes[url]
    if (route instanceof Error) throw route
    if (typeof route === 'string') return { ok: true, status: 200, text: async () => route }
    if (typeof route === 'object' && 'json' in route) return { ok: true, status: 200, text: async () => JSON.stringify(route.json) }
    return { ok: false, status: /** @type {{ status: number }} */ (route).status, text: async () => '' }
  }
  return { fetchImpl, calls }
}

test('parseShasums256 reads the binary marker and plain spacing, and only the requested file', () => {
  const text = [
    `${'a'.repeat(64)} *electron-v40.1.0-darwin-arm64.zip`,
    `${'b'.repeat(64)}  electron-v40.1.0-darwin-x64.zip`,
    `${'c'.repeat(64)} *electron-v40.1.0-win32-x64.zip`,
  ].join('\r\n')
  assert.equal(parseShasums256(text, 'electron-v40.1.0-darwin-arm64.zip'), 'a'.repeat(64))
  assert.equal(parseShasums256(text, 'electron-v40.1.0-darwin-x64.zip'), 'b'.repeat(64))
  assert.equal(parseShasums256(text, 'electron-v40.1.0-linux-x64.zip'), null)
  assert.equal(parseShasums256('', 'electron-v40.1.0-darwin-arm64.zip'), null)
})

test('sha256File hashes the artifact bytes', async () => {
  const { archive, sha } = fixture('electron zip bytes')
  assert.equal(await sha256File(archive), sha)
})

test('fetchOfficialShasums asks the official release URL first, without credentials', async () => {
  const { fetchImpl, calls } = fakeFetch({ [DIRECT]: 'manifest text' })
  const result = await fetchOfficialShasums(VERSION, { fetchImpl })
  assert.equal(result.url, DIRECT)
  assert.equal(result.text, 'manifest text')
  assert.deepEqual(calls, [{ url: DIRECT, headers: {} }])
})

test('a matching official hash verifies and caches the manifest for later builds', async () => {
  const { root, archive, filename, sha } = fixture('good zip')
  const { fetchImpl } = fakeFetch({ [DIRECT]: manifest(filename, sha) })
  const result = await verifyElectronArtifact(archive, { version: VERSION, cacheDir: root, fetchImpl, log: quiet })
  assert.equal(result.verified, true)
  assert.equal(result.source, 'official')
  assert.equal(result.sha256, sha)
  assert.equal(fs.readFileSync(path.join(root, `SHASUMS256-v${VERSION}.txt`), 'utf8'), manifest(filename, sha))
})

test('a mismatching hash deletes the poisoned archive and fails the build', async () => {
  const { root, archive, filename } = fixture('substituted zip')
  const { fetchImpl } = fakeFetch({ [DIRECT]: manifest(filename, 'f'.repeat(64)) })
  await assert.rejects(
    verifyElectronArtifact(archive, { version: VERSION, cacheDir: root, fetchImpl, log: quiet }),
    /does not match the official release hash/,
  )
  assert.equal(fs.existsSync(archive), false)
})

test('an artifact the official manifest does not list fails instead of passing unverified', async () => {
  const { root, archive } = fixture('unknown artifact')
  const { fetchImpl } = fakeFetch({ [DIRECT]: manifest('electron-v40.1.0-linux-x64.zip', 'e'.repeat(64)) })
  await assert.rejects(
    verifyElectronArtifact(archive, { version: VERSION, cacheDir: root, fetchImpl, log: quiet }),
    /does not list/,
  )
  assert.equal(fs.existsSync(archive), true)
})

test('the GitHub releases API is the fallback when the release download URL is unreachable', async () => {
  const { root, archive, filename, sha } = fixture('api-path zip')
  const assetUrl = `${OFFICIAL_API_BASE}/repos/electron/electron/releases/assets/123`
  const { fetchImpl, calls } = fakeFetch({
    [DIRECT]: new Error('connect ETIMEDOUT'),
    [RELEASE_API]: { json: { assets: [{ name: 'electron-v40.1.0-darwin-arm64.zip', url: 'https://example.invalid/unrelated' }, { name: 'SHASUMS256.txt', url: assetUrl }] } },
    [assetUrl]: manifest(filename, sha),
  })
  const result = await verifyElectronArtifact(archive, {
    version: VERSION, cacheDir: root, fetchImpl, log: quiet, token: 'test-token',
  })
  assert.equal(result.verified, true)
  assert.equal(result.source, 'official')
  const byUrl = new Map(calls.map(call => [call.url, call.headers]))
  assert.equal(byUrl.get(DIRECT)?.Authorization, undefined, 'the release CDN never receives the token')
  assert.equal(byUrl.get(RELEASE_API)?.Authorization, 'Bearer test-token')
  assert.equal(byUrl.get(assetUrl)?.Accept, 'application/octet-stream')
})

test('both official channels down: the build runs but reports the artifact unverified', async () => {
  const { root, archive } = fixture('cached zip')
  const { fetchImpl } = fakeFetch({
    [DIRECT]: new Error('getaddrinfo ENOTFOUND github.com'),
    [RELEASE_API]: { status: 403 },
  })
  const result = await verifyElectronArtifact(archive, { version: VERSION, cacheDir: root, fetchImpl, log: quiet })
  assert.equal(result.verified, false)
  assert.equal(result.source, null)
  assert.match(String(result.reason), /unreachable/)
  assert.equal(fs.existsSync(archive), true)
})

test('strict mode refuses to build on an unverifiable artifact', async () => {
  const { root, archive } = fixture('cached zip')
  const { fetchImpl } = fakeFetch({
    [DIRECT]: new Error('connect ECONNREFUSED'),
    [RELEASE_API]: { status: 403 },
  })
  await assert.rejects(
    verifyElectronArtifact(archive, { version: VERSION, cacheDir: root, fetchImpl, log: quiet, requireOfficial: true }),
    /unreachable/,
  )
})

test('an offline build verifies against the manifest a previous online build cached', async () => {
  const { root, archive, filename, sha } = fixture('offline zip')
  fs.writeFileSync(path.join(root, `SHASUMS256-v${VERSION}.txt`), manifest(filename, sha))
  const { fetchImpl } = fakeFetch({
    [DIRECT]: new Error('connect ECONNREFUSED'),
    [RELEASE_API]: { status: 503 },
  })
  const result = await verifyElectronArtifact(archive, { version: VERSION, cacheDir: root, fetchImpl, log: quiet })
  assert.equal(result.verified, true)
  assert.equal(result.source, 'cache')
})

test('an offline build whose cached manifest disagrees with the archive still fails and purges', async () => {
  const { root, archive } = fixture('tampered offline zip')
  fs.writeFileSync(path.join(root, `SHASUMS256-v${VERSION}.txt`), manifest(path.basename(archive), 'd'.repeat(64)))
  const { fetchImpl } = fakeFetch({
    [DIRECT]: new Error('connect ECONNREFUSED'),
    [RELEASE_API]: { status: 503 },
  })
  await assert.rejects(
    verifyElectronArtifact(archive, { version: VERSION, cacheDir: root, fetchImpl, log: quiet }),
    /does not match the official release hash/,
  )
  assert.equal(fs.existsSync(archive), false)
})
