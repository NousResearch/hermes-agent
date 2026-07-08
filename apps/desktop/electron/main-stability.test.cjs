'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ELECTRON_DIR = __dirname

function readElectronFile(name) {
  return fs.readFileSync(path.join(ELECTRON_DIR, name), 'utf8').replace(/\r\n/g, '\n')
}

// ---------------------------------------------------------------------------
// A1: the revalidate liveness probe must require a streak of consecutive
// failures (default 3) before dropping the cached remote connection, and must
// give each probe 8s (env-tunable) instead of the old 2.5s hair trigger.
// ---------------------------------------------------------------------------

test('revalidate probe uses the tunable timeout, not the old 2.5s hair trigger', () => {
  const source = readElectronFile('main.cjs')

  assert.match(
    source,
    /HERMES_DESKTOP_REVALIDATE_TIMEOUT_MS\) \|\| 8_000/,
    'probe timeout should default to 8s with an env override'
  )
  assert.match(
    source,
    /HERMES_DESKTOP_REVALIDATE_FAILURES\) \|\| 3/,
    'failure streak should default to 3 with an env override'
  )
  assert.doesNotMatch(
    source,
    /api\/status`, \{ timeoutMs: 2_500 \}/,
    'no /api/status probe should still use the 2.5s timeout'
  )

  const probeCount = (source.match(/timeoutMs: REVALIDATE_PROBE_TIMEOUT_MS/g) || []).length
  assert.ok(
    probeCount >= 2,
    `primary + pooled probes should both use REVALIDATE_PROBE_TIMEOUT_MS (found ${probeCount})`
  )
})

test('revalidate drops the cached connection only after a failure streak', () => {
  const source = readElectronFile('main.cjs')

  assert.match(source, /require\('\.\/liveness-streak\.cjs'\)/, 'main.cjs should use the liveness-streak module')
  assert.match(
    source,
    /createLivenessStreak\(\{ threshold: REVALIDATE_FAILURE_STREAK \}\)/,
    'streak tracker should be built from the tunable threshold'
  )

  const fnStart = source.indexOf('async function revalidatePrimaryConnection(')
  assert.notEqual(fnStart, -1, 'revalidatePrimaryConnection function not found')
  const fnBody = source.slice(fnStart, fnStart + 2400)

  assert.match(fnBody, /primaryLivenessStreak\.recordSuccess\(\)/, 'a successful probe must reset the streak')
  assert.match(fnBody, /primaryLivenessStreak\.recordFailure\(base\)/, 'a failed probe must feed the streak')

  // The drop log + reset must be gated behind the streak verdict.
  const dropIdx = fnBody.indexOf('failed liveness probe; dropping stale connection')
  const guardIdx = fnBody.indexOf('if (!failure.drop)')
  assert.ok(guardIdx !== -1, 'drop must be guarded on the streak verdict')
  assert.ok(dropIdx > guardIdx, 'drop log/reset must come after the non-drop early return')

  // Intermediate failures log at most once per streak.
  assert.match(fnBody, /failure\.firstOfStreak/, 'intermediate failures should log only on the first of a streak')
})

// ---------------------------------------------------------------------------
// A2: only HTTP 401/403 from the ws-ticket mint means "session expired";
// transport failures must rethrow as retryable (no needsOauthLogin).
// ---------------------------------------------------------------------------

test('buildRemoteConnection classifies ticket-mint failures by statusCode', () => {
  const source = readElectronFile('main.cjs')

  const fnStart = source.indexOf('async function buildRemoteConnection(')
  assert.notEqual(fnStart, -1, 'buildRemoteConnection function not found')
  const fnBody = source.slice(fnStart, source.indexOf('async function resolveRemoteBackend('))

  assert.match(
    fnBody,
    /const status = Number\(error\?\.statusCode\)/,
    'the mint catch must inspect the underlying statusCode'
  )
  assert.match(fnBody, /status !== 401 && status !== 403/, 'only 401/403 may be classified as an expired session')
  assert.match(
    fnBody,
    /Could not reach the remote Hermes gateway/,
    'non-auth failures must surface as a retryable unreachable-backend error'
  )

  // The retryable branch must not flag needsOauthLogin: the only two
  // needsOauthLogin assignments in the OAuth path are the not-signed-in
  // early-out and the confirmed-401/403 expiry.
  const flagCount = (fnBody.match(/err\.needsOauthLogin = true/g) || []).length
  assert.equal(flagCount, 2, `expected exactly 2 needsOauthLogin assignments, found ${flagCount}`)
  const retryableIdx = fnBody.indexOf('Could not reach the remote Hermes gateway')
  const expiredIdx = fnBody.indexOf('Your remote gateway session has expired')
  assert.ok(retryableIdx < expiredIdx, 'retryable rethrow must short-circuit before the expiry classification')
})

// ---------------------------------------------------------------------------
// A3a: in app-global remote mode, profiles without their own remote override
// must NOT register reapable pool entries — they ride the shared descriptor.
// ---------------------------------------------------------------------------

test('ensureBackend short-circuits global-remote profiles past the pool', () => {
  const source = readElectronFile('main.cjs')

  const fnStart = source.indexOf('async function ensureBackend(')
  assert.notEqual(fnStart, -1, 'ensureBackend function not found')
  const fnBody = source.slice(fnStart, fnStart + 2200)

  const shortCircuitIdx = fnBody.indexOf('globalRemoteActive() && !profileHasRemoteOverride(key)')
  const poolIdx = fnBody.indexOf('backendPool.get(key)')
  assert.notEqual(shortCircuitIdx, -1, 'ensureBackend must short-circuit global-remote profiles')
  assert.ok(shortCircuitIdx < poolIdx, 'short-circuit must run before any pool registration/lookup')

  // The shared descriptor must still be tagged with the profile so fs/git
  // REST calls and ?profile= injection keep working.
  assert.match(fnBody, /\{ \.\.\.conn, profile: key \}/, 'shared descriptor must carry the requested profile')
})

// ---------------------------------------------------------------------------
// A3c: revalidate must also probe pooled REMOTE descriptors (process === null)
// and delete dead ones.
// ---------------------------------------------------------------------------

test('revalidate probes pooled remote descriptors and drops dead ones', () => {
  const source = readElectronFile('main.cjs')

  const fnStart = source.indexOf('async function revalidatePooledRemoteBackends(')
  assert.notEqual(fnStart, -1, 'revalidatePooledRemoteBackends function not found')
  const fnBody = source.slice(fnStart, fnStart + 1200)

  assert.match(fnBody, /!entry\.process && entry\.remoteBaseUrl/, 'only process-less remote descriptors are probed')
  assert.match(fnBody, /stopPoolBackend\(profile\)/, 'a dead pooled remote must be dropped from the pool')

  // The handler runs both checks.
  assert.match(
    source,
    /Promise\.all\(\[revalidatePrimaryConnection\(\), revalidatePooledRemoteBackends\(\)\]\)/,
    'the revalidate IPC handler must run primary and pooled checks together'
  )

  // spawnPoolBackend records the base URL for later probing.
  assert.match(source, /entry\.remoteBaseUrl = remote\.baseUrl/, 'remote pool entries must record their base URL')
})

// ---------------------------------------------------------------------------
// A4: mergeRemoteProfileSessions must use the authMode-aware helper for the
// primary list (raw token-header fetchJson 401s against an OAuth primary).
// ---------------------------------------------------------------------------

test('mergeRemoteProfileSessions routes the primary list through fetchJsonForProfile', () => {
  const source = readElectronFile('main.cjs')

  const fnStart = source.indexOf('async function mergeRemoteProfileSessions(')
  assert.notEqual(fnStart, -1, 'mergeRemoteProfileSessions function not found')
  const fnBody = source.slice(fnStart, fnStart + 2400)

  assert.match(
    fnBody,
    /fetchJsonForProfile\(null, `\/api\/profiles\/sessions\?\$\{searchParams\}`\)/,
    'primary list must go through the authMode-aware helper'
  )
  assert.doesNotMatch(
    fnBody,
    /fetchJson\(`\$\{primary\.baseUrl\}/,
    'the raw token-header fetchJson against the primary must be gone'
  )
})

// ---------------------------------------------------------------------------
// A5: global error handlers + no unhandled loadURL promises.
// ---------------------------------------------------------------------------

test('main process installs uncaughtException/unhandledRejection forensics', () => {
  const source = readElectronFile('main.cjs')

  const uncaughtIdx = source.indexOf("process.on('uncaughtException'")
  const rejectionIdx = source.indexOf("process.on('unhandledRejection'")
  assert.notEqual(uncaughtIdx, -1, 'uncaughtException handler not found')
  assert.notEqual(rejectionIdx, -1, 'unhandledRejection handler not found')

  // Both handlers must record AND flush synchronously so the log survives.
  for (const [name, idx] of [
    ['uncaughtException', uncaughtIdx],
    ['unhandledRejection', rejectionIdx]
  ]) {
    const body = source.slice(idx, idx + 400)
    assert.match(body, /rememberLog\(/, `${name} handler must rememberLog`)
    assert.match(body, /flushDesktopLogBufferSync\(\)/, `${name} handler must flush the log buffer`)
  }
})

test('every loadURL call handles rejection', () => {
  const source = readElectronFile('main.cjs')

  // Every `.loadURL(...)` call expression must be followed by a `.catch(` (or
  // an options-object continuation that itself ends in `.catch(`). Walk each
  // occurrence and scan past its balanced parens for the catch.
  const sites = []
  const re = /\.loadURL\(/g
  let m
  while ((m = re.exec(source)) !== null) sites.push(m.index)
  assert.ok(sites.length >= 5, `expected at least 5 loadURL sites, found ${sites.length}`)

  for (const site of sites) {
    let depth = 0
    let i = site + '.loadURL'.length
    do {
      const ch = source[i]
      if (ch === '(') depth += 1
      else if (ch === ')') depth -= 1
      i += 1
    } while (depth > 0 && i < source.length)

    const after = source.slice(i, i + 40).replace(/\s+/g, '')
    const line = source.slice(0, site).split('\n').length
    assert.ok(
      after.startsWith('.catch(') || after.startsWith(',{') || after.startsWith('.then('),
      `loadURL at line ${line} must chain .catch (got: ${after.slice(0, 20)})`
    )
    if (after.startsWith(',{')) continue // multi-arg form handled below
    if (after.startsWith('.then(')) {
      assert.match(source.slice(i, i + 400), /\.catch\(/, `loadURL at line ${line} .then chain must end in .catch`)
    }
  }

  // The multi-argument form (link-title window) already catches; assert it
  // stays that way rather than special-casing silently.
  assert.match(source, /\.loadURL\(rawUrl, \{[\s\S]{0,200}\}\)\s*\.catch\(/, 'multi-arg loadURL must keep its .catch')
})
