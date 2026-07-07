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
// prepareProfileDeleteRequest must return the torn-down profile name so the
// caller can skip ensureBackend for that profile (issue #52279).
// ---------------------------------------------------------------------------

test('prepareProfileDeleteRequest returns the torn-down profile name', () => {
  const source = readElectronFile('main.cjs')

  // Locate the function definition and its closing brace.
  const fnStart = source.indexOf('async function prepareProfileDeleteRequest(')
  assert.notEqual(fnStart, -1, 'prepareProfileDeleteRequest function not found')

  // The function must contain "return profile" (pool and primary paths).
  const fnBody = source.slice(fnStart, fnStart + 800)
  const returnProfileCount = (fnBody.match(/return profile/g) || []).length
  assert.ok(
    returnProfileCount >= 2,
    `expected at least 2 "return profile" statements (primary + pool paths), found ${returnProfileCount}`
  )

  // The early-exit guard must return null (not void/undefined).
  assert.match(fnBody, /return null/, 'early-exit guard should return null, not undefined')
})

test('hermes:api handler routes profile-delete requests to the primary backend', () => {
  const source = readElectronFile('main.cjs')

  // The handler must capture prepareProfileDeleteRequest's return value.
  assert.match(
    source,
    /const tornDownProfile = await prepareProfileDeleteRequest\(request\)/,
    'handler should capture the return value of prepareProfileDeleteRequest'
  )

  // The handler must use the return value to skip ensureBackend for the
  // torn-down profile, routing to the primary (null) instead.
  assert.match(
    source,
    /const routeProfile = tornDownProfile \? null : profile/,
    'handler should route to primary backend when a profile was just torn down'
  )

  // ensureBackend must be called with the conditional route profile.
  assert.match(
    source,
    /const connection = await ensureBackend\(routeProfile\)/,
    'handler should pass routeProfile (not raw profile) to ensureBackend'
  )
})

test('hermes:api handler restarts a timed-out local pooled profile backend once', () => {
  const source = readElectronFile('main.cjs')

  assert.match(source, /function isHermesBackendTimeout\(error\)/, 'missing timeout classifier')
  assert.match(
    source,
    /function shouldRetryLocalPooledBackend\(profile, connection, error\)/,
    'missing local pooled backend retry guard'
  )
  assert.match(
    source,
    /connection\?\.mode === 'local' && connection\?\.source === 'local'/,
    'retry guard should apply only to local pooled backends'
  )
  assert.match(
    source,
    /stopPoolBackend\(key\)[\s\S]*const restarted = await ensureBackend\(key\)[\s\S]*return send\(restarted\)/,
    'handler should stop, respawn, and retry the local profile backend'
  )
  assert.match(
    source,
    /event: 'profile\.timeout\.restart'/,
    'restart should emit a diagnostic event for crash forensics'
  )
})
