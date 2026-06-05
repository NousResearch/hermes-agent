const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')
const { pathToFileURL } = require('node:url')

const {
  DEFAULT_FETCH_TIMEOUT_MS,
  REMOTE_REVALIDATE_FAILURE_THRESHOLD,
  REVALIDATE_PROBE_TIMEOUT_MS,
  REVALIDATE_STREAK_WINDOW_MS,
  encryptDesktopSecret,
  isFreshRevalidateEpisode,
  probeBackendAlive,
  resolveReadableFileForIpc,
  resolveTimeoutMs,
  sensitiveFileBlockReason,
  shouldRebuildStaleConnection
} = require('./hardening.cjs')

test('resolveTimeoutMs falls back to defaults and accepts overrides', () => {
  assert.equal(resolveTimeoutMs(undefined), DEFAULT_FETCH_TIMEOUT_MS)
  assert.equal(resolveTimeoutMs(0), DEFAULT_FETCH_TIMEOUT_MS)
  assert.equal(resolveTimeoutMs(-25), DEFAULT_FETCH_TIMEOUT_MS)
  assert.equal(resolveTimeoutMs('2750'), 2750)
})

test('encryptDesktopSecret requires available secure storage', () => {
  assert.equal(
    encryptDesktopSecret('', { isEncryptionAvailable: () => true, encryptString: () => Buffer.alloc(0) }),
    null
  )

  assert.throws(
    () => encryptDesktopSecret('token', { isEncryptionAvailable: () => false, encryptString: () => Buffer.alloc(0) }),
    /Secure token storage is unavailable/
  )
})

test('encryptDesktopSecret stores safeStorage base64 payload', () => {
  const secret = encryptDesktopSecret('token-123', {
    isEncryptionAvailable: () => true,
    encryptString: value => Buffer.from(`enc:${value}`, 'utf8')
  })

  assert.deepEqual(secret, {
    encoding: 'safeStorage',
    value: Buffer.from('enc:token-123', 'utf8').toString('base64')
  })
})

test('sensitiveFileBlockReason blocks obvious secret file patterns', () => {
  assert.match(String(sensitiveFileBlockReason('/tmp/.env')), /\.env/)
  assert.equal(sensitiveFileBlockReason('/tmp/.env.example'), null)
  assert.match(String(sensitiveFileBlockReason('/Users/me/.ssh/id_ed25519')), /SSH/)
  assert.match(String(sensitiveFileBlockReason('/tmp/server-cert.pem')), /\.pem/)
})

test('resolveReadableFileForIpc validates existence type size and sensitivity', async t => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-hardening-'))
  t.after(() => fs.rmSync(tempDir, { recursive: true, force: true }))

  const textPath = path.join(tempDir, 'notes.txt')
  fs.writeFileSync(textPath, 'hello world', 'utf8')

  const fromRelative = await resolveReadableFileForIpc('notes.txt', {
    baseDir: tempDir,
    maxBytes: 256,
    purpose: 'File preview'
  })
  assert.equal(fromRelative.resolvedPath, textPath)
  assert.equal(fromRelative.stat.size, 11)

  const fromFileUrl = await resolveReadableFileForIpc(pathToFileURL(textPath).toString(), {
    purpose: 'File preview'
  })
  assert.equal(fromFileUrl.resolvedPath, textPath)

  await assert.rejects(
    resolveReadableFileForIpc('missing.txt', {
      baseDir: tempDir,
      purpose: 'Text preview'
    }),
    /file does not exist/
  )

  const nestedDir = path.join(tempDir, 'directory')
  fs.mkdirSync(nestedDir)
  await assert.rejects(
    resolveReadableFileForIpc(nestedDir, {
      purpose: 'Text preview'
    }),
    /path points to a directory/
  )

  const largePath = path.join(tempDir, 'large.txt')
  fs.writeFileSync(largePath, 'x'.repeat(40), 'utf8')
  await assert.rejects(
    resolveReadableFileForIpc(largePath, {
      maxBytes: 8,
      purpose: 'File preview'
    }),
    /file is too large/
  )

  const envPath = path.join(tempDir, '.env')
  fs.writeFileSync(envPath, 'SECRET_TOKEN=123', 'utf8')
  await assert.rejects(
    resolveReadableFileForIpc(envPath, {
      purpose: 'File preview'
    }),
    /blocked for sensitive file/
  )

  const envTemplatePath = path.join(tempDir, '.env.example')
  fs.writeFileSync(envTemplatePath, 'EXAMPLE_TOKEN=value', 'utf8')
  const envTemplate = await resolveReadableFileForIpc(envTemplatePath, {
    purpose: 'File preview'
  })
  assert.equal(envTemplate.resolvedPath, envTemplatePath)
})

// ── Sleep/wake reconnect: cached-connection liveness revalidation ────────────

test('probeBackendAlive returns true when /api/status responds', async () => {
  const calls = []
  const fetchJson = async (url, opts) => {
    calls.push({ url, opts })
    return { ok: true }
  }

  const alive = await probeBackendAlive({ baseUrl: 'https://gw.example.com' }, fetchJson, { timeoutMs: 2500 })

  assert.equal(alive, true)
  assert.equal(calls.length, 1)
  // Trailing slash on baseUrl must not double up before /api/status.
  assert.equal(calls[0].url, 'https://gw.example.com/api/status')
  assert.equal(calls[0].opts.timeoutMs, 2500)
})

test('probeBackendAlive normalizes a trailing slash on baseUrl', async () => {
  let seen = null
  const fetchJson = async url => {
    seen = url
    return null
  }

  await probeBackendAlive({ baseUrl: 'http://127.0.0.1:8787/' }, fetchJson)
  assert.equal(seen, 'http://127.0.0.1:8787/api/status')
})

test('probeBackendAlive returns false (never throws) when the probe rejects', async () => {
  const fetchJson = async () => {
    throw new Error('ECONNREFUSED')
  }

  const alive = await probeBackendAlive({ baseUrl: 'https://gw.example.com' }, fetchJson)
  assert.equal(alive, false)
})

test('probeBackendAlive returns false for a missing baseUrl or fetcher', async () => {
  const ok = async () => ({})
  assert.equal(await probeBackendAlive(null, ok), false)
  assert.equal(await probeBackendAlive({ baseUrl: '' }, ok), false)
  assert.equal(await probeBackendAlive({ baseUrl: 'https://gw.example.com' }, undefined), false)
})

test('probeBackendAlive defaults the probe timeout to REVALIDATE_PROBE_TIMEOUT_MS', async () => {
  let seenTimeout
  const fetchJson = async (_url, opts) => {
    seenTimeout = opts.timeoutMs
    return {}
  }

  await probeBackendAlive({ baseUrl: 'https://gw.example.com' }, fetchJson)
  assert.equal(seenTimeout, REVALIDATE_PROBE_TIMEOUT_MS)
})

test('shouldRebuildStaleConnection only tears down a remote past the failure threshold', () => {
  // A live backend is never rebuilt, regardless of streak.
  assert.equal(
    shouldRebuildStaleConnection({ alive: true, mode: 'remote', failureStreak: 99 }),
    false
  )

  // A local backend is never rebuilt here — it self-heals via its child 'exit'
  // handler, so a probe miss means "WS not reattached yet", not "backend dead".
  assert.equal(
    shouldRebuildStaleConnection({ alive: false, mode: 'local', failureStreak: 99 }),
    false
  )

  // A single remote miss is ignored (rides out transient post-wake blips)…
  assert.equal(
    shouldRebuildStaleConnection({ alive: false, mode: 'remote', failureStreak: 1 }),
    false
  )

  // …but the second consecutive miss crosses the threshold and rebuilds.
  assert.equal(
    shouldRebuildStaleConnection({
      alive: false,
      mode: 'remote',
      failureStreak: REMOTE_REVALIDATE_FAILURE_THRESHOLD
    }),
    true
  )

  // Unknown / undefined mode (e.g. a rejected cache) is not eligible.
  assert.equal(
    shouldRebuildStaleConnection({ alive: false, mode: undefined, failureStreak: 5 }),
    false
  )
})

test('REMOTE_REVALIDATE_FAILURE_THRESHOLD requires at least two misses', () => {
  assert.ok(REMOTE_REVALIDATE_FAILURE_THRESHOLD >= 2)
})

test('isFreshRevalidateEpisode scopes the failure streak to one reconnect episode', () => {
  // First-ever failure (no prior timestamp) always starts a fresh episode.
  assert.equal(isFreshRevalidateEpisode(1_000, 0), true)

  // A miss within the window continues the same episode (streak accumulates).
  assert.equal(isFreshRevalidateEpisode(10_000, 10_000 - (REVALIDATE_STREAK_WINDOW_MS - 1)), false)

  // A miss past the window is a new episode (streak resets) — this is what stops
  // a stale miss from an earlier, since-recovered episode pre-loading the counter.
  assert.equal(isFreshRevalidateEpisode(10_000, 10_000 - (REVALIDATE_STREAK_WINDOW_MS + 1)), true)

  // The window comfortably exceeds the renderer's 15s backoff cap so two
  // genuinely-consecutive backoff-paced misses always fall inside it.
  assert.ok(REVALIDATE_STREAK_WINDOW_MS > 15_000)
})
