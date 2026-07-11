'use strict'

const assert = require('node:assert/strict')
const test = require('node:test')

const {
  STATUS_START,
  STATUS_SUCCESS,
  STATUS_FAILURE,
  VALID_STATUSES,
  DEFAULT_API_BASE,
  TELEMETRY_PATH,
  FIELD_MAX_LENGTHS,
  isTelemetryDisabled,
  telemetryEndpoint,
  buildPayload,
  normalizeDesktopPlatform,
  classifyErrorCategory,
  buildErrorCode,
  sendDesktopTelemetry,
  fireTelemetry
} = require('./apexnodes-telemetry.cjs')

// ---------------------------------------------------------------------------
// isTelemetryDisabled / telemetryEndpoint
// ---------------------------------------------------------------------------

test('isTelemetryDisabled: recognizes the documented off switch + lenient synonyms', () => {
  assert.equal(isTelemetryDisabled({ APEXNODES_TELEMETRY: 'off' }), true)
  assert.equal(isTelemetryDisabled({ APEXNODES_TELEMETRY: 'OFF' }), true, 'case-insensitive')
  assert.equal(isTelemetryDisabled({ APEXNODES_TELEMETRY: ' off ' }), true, 'trims whitespace')
  assert.equal(isTelemetryDisabled({ APEXNODES_TELEMETRY: '0' }), true)
  assert.equal(isTelemetryDisabled({ APEXNODES_TELEMETRY: 'false' }), true)
  assert.equal(isTelemetryDisabled({ APEXNODES_TELEMETRY: 'disabled' }), true)
})

test('isTelemetryDisabled: enabled by default (unset or any other value)', () => {
  assert.equal(isTelemetryDisabled({}), false)
  assert.equal(isTelemetryDisabled({ APEXNODES_TELEMETRY: 'on' }), false)
  assert.equal(isTelemetryDisabled({ APEXNODES_TELEMETRY: '1' }), false)
})

test('telemetryEndpoint: defaults to api.apex-nodes.com/api/v1/desktop/telemetry', () => {
  assert.equal(telemetryEndpoint({}), `${DEFAULT_API_BASE}${TELEMETRY_PATH}`)
})

test('telemetryEndpoint: honors the shared APEXNODES_API_BASE override (same var apex-managed.cjs uses)', () => {
  assert.equal(
    telemetryEndpoint({ APEXNODES_API_BASE: 'https://staging.example.com/' }),
    'https://staging.example.com/api/v1/desktop/telemetry'
  )
})

// ---------------------------------------------------------------------------
// normalizeDesktopPlatform
// ---------------------------------------------------------------------------

test('normalizeDesktopPlatform: maps node platform strings to the win/mac/linux vocabulary', () => {
  assert.equal(normalizeDesktopPlatform('darwin'), 'mac')
  assert.equal(normalizeDesktopPlatform('win32'), 'win')
  assert.equal(normalizeDesktopPlatform('linux'), 'linux')
  assert.equal(normalizeDesktopPlatform('mac'), 'mac', 'already-normalized value passes through')
  assert.equal(normalizeDesktopPlatform('win'), 'win', 'already-normalized value passes through')
  assert.equal(normalizeDesktopPlatform(''), 'linux', 'empty degrades to linux rather than throwing')
})

// ---------------------------------------------------------------------------
// buildPayload — allow-list + truncation
// ---------------------------------------------------------------------------

test('buildPayload: only the seven anonymous fields ever survive', () => {
  const payload = buildPayload({
    platform: 'win',
    arch: 'x64',
    app_version: '0.16.7',
    runtime_key: 'v2026.7.1',
    stage: 'repository',
    status: 'success',
    error_code: null,
    // None of these must ever appear in the request body.
    user_id: 'u_123',
    email: 'kael@example.com',
    ip: '1.2.3.4',
    session_token: 'secret',
    path: '/Users/kael/.hermes'
  })
  assert.deepEqual(payload, {
    platform: 'win',
    arch: 'x64',
    app_version: '0.16.7',
    runtime_key: 'v2026.7.1',
    stage: 'repository',
    status: 'success'
  })
})

test('buildPayload: drops undefined/null/empty-string optional fields entirely (no key at all)', () => {
  const payload = buildPayload({ platform: 'mac', stage: 'venv', status: 'start', arch: undefined, error_code: '' })
  assert.deepEqual(payload, { platform: 'mac', stage: 'venv', status: 'start' })
  assert.equal('arch' in payload, false)
  assert.equal('error_code' in payload, false)
})

test('buildPayload: truncates any field over its cloud-schema max_length', () => {
  const longError = 'x'.repeat(500)
  const payload = buildPayload({ platform: 'win', stage: 'dependencies', status: 'failure', error_code: longError })
  assert.equal(payload.error_code.length, FIELD_MAX_LENGTHS.error_code)
  assert.equal(payload.error_code, longError.slice(0, FIELD_MAX_LENGTHS.error_code))
})

// ---------------------------------------------------------------------------
// classifyErrorCategory / buildErrorCode — no paths, no user info, ever
// ---------------------------------------------------------------------------

test('classifyErrorCategory: buckets common Node/install error shapes, never echoes the raw message', () => {
  const cases = [
    [new Error('Timed out connecting to Hermes backend after 10000ms'), 'timeout'],
    [new Error("ENOENT: no such file or directory, open '/Users/kael/.hermes/foo'"), 'not_found'],
    [new Error("EACCES: permission denied, mkdir '/Users/kael/.apexnodes'"), 'permission'],
    [new Error('getaddrinfo ENOTFOUND api.apex-nodes.com'), 'network'],
    [new Error('sha256 mismatch on 3 files'), 'checksum_mismatch'],
    [new Error('exit code 1'), 'exit_nonzero'],
    [new Error('install.sh --stage venv produced no JSON result frame (exit=1)'), 'protocol'],
    [new Error('runtime bundle needs desktop >= 0.18.0 (this shell is 0.17.0)'), 'unknown'],
    [{ code: 'min_desktop_version', message: 'min_desktop_version too old' }, 'incompatible'],
    [new Error('bootstrap cancelled by user'), 'cancelled'],
    [null, 'unknown'],
    ['', 'unknown']
  ]
  for (const [err, want] of cases) {
    assert.equal(classifyErrorCategory(err), want, `expected ${want} for ${err && err.message}`)
  }
})

test('classifyErrorCategory: a message carrying an absolute path never leaks the path into the category', () => {
  const category = classifyErrorCategory(new Error('ENOENT: /Users/kael/Secret Project/install.sh not found'))
  assert.equal(category, 'not_found')
  assert.ok(!category.includes('/'), 'category bucket never contains path separators')
  assert.ok(!category.includes('kael'), 'category bucket never contains the username')
})

test('buildErrorCode: "<stage>:<category>" shape, capped to error_code max_length', () => {
  assert.equal(buildErrorCode('repository', new Error('ECONNRESET')), 'repository:network')
  assert.equal(buildErrorCode('system-packages', new Error('exit code 1')), 'system-packages:exit_nonzero')
  const longStage = 'a'.repeat(200)
  assert.equal(buildErrorCode(longStage, new Error('boom')).length, FIELD_MAX_LENGTHS.error_code)
})

// ---------------------------------------------------------------------------
// sendDesktopTelemetry — fake transport only, NEVER the real network
// ---------------------------------------------------------------------------

function baseEvent(overrides = {}) {
  return { platform: 'win', arch: 'x64', stage: 'repository', status: STATUS_SUCCESS, ...overrides }
}

test('sendDesktopTelemetry: kill switch short-circuits before any transport call', async () => {
  let called = false
  const result = await sendDesktopTelemetry(baseEvent(), {
    env: { APEXNODES_TELEMETRY: 'off' },
    _post: async () => {
      called = true
      return { ok: true }
    }
  })
  assert.equal(result.ok, false)
  assert.equal(result.skipped, 'disabled')
  assert.equal(called, false, 'the transport must never be invoked once disabled')
})

test('sendDesktopTelemetry: fake-fetch success path resolves ok:true and posts the allow-listed body', async () => {
  let seenUrl = null
  let seenPayload = null
  const result = await sendDesktopTelemetry(
    baseEvent({ error_code: null, runtime_key: 'v2026.7.1', extra_field: 'must-not-appear' }),
    {
      env: {},
      _post: async (url, payload) => {
        seenUrl = url
        seenPayload = payload
        return { ok: true, status: 201 }
      }
    }
  )
  assert.equal(result.ok, true)
  assert.equal(seenUrl, `${DEFAULT_API_BASE}${TELEMETRY_PATH}`)
  assert.deepEqual(seenPayload, {
    platform: 'win',
    arch: 'x64',
    stage: 'repository',
    status: 'success',
    runtime_key: 'v2026.7.1'
  })
})

test('sendDesktopTelemetry: fake-fetch timeout resolves ok:false, never rejects', async () => {
  const result = await sendDesktopTelemetry(baseEvent(), {
    env: {},
    _post: async () => ({ ok: false, error: 'timeout' })
  })
  assert.equal(result.ok, false)
  assert.equal(result.error, 'timeout')
})

test('sendDesktopTelemetry: fake-fetch offline (rejecting transport) still resolves, never rejects/throws', async () => {
  await assert.doesNotReject(async () => {
    const result = await sendDesktopTelemetry(baseEvent(), {
      env: {},
      _post: async () => {
        throw new Error('getaddrinfo ENOTFOUND api.apex-nodes.com')
      }
    })
    assert.equal(result.ok, false)
    assert.match(result.error, /ENOTFOUND/)
  })
})

test('sendDesktopTelemetry: a synchronously-throwing transport still resolves, never throws', async () => {
  const result = await sendDesktopTelemetry(baseEvent(), {
    env: {},
    _post: () => {
      throw new Error('boom')
    }
  })
  assert.equal(result.ok, false)
  assert.equal(result.error, 'boom')
})

test('sendDesktopTelemetry: rejects invalid event shapes without ever calling the transport', async () => {
  let called = false
  const post = async () => {
    called = true
    return { ok: true }
  }
  const missingStage = await sendDesktopTelemetry({ platform: 'win', status: STATUS_START }, { env: {}, _post: post })
  const badStatus = await sendDesktopTelemetry(baseEvent({ status: 'bogus' }), { env: {}, _post: post })
  const missingPlatform = await sendDesktopTelemetry({ stage: 'x', status: STATUS_START }, { env: {}, _post: post })
  assert.equal(missingStage.ok, false)
  assert.equal(badStatus.ok, false)
  assert.equal(missingPlatform.ok, false)
  assert.equal(called, false)
})

test('VALID_STATUSES matches the tri-state vocabulary the cloud Pydantic validator enforces', () => {
  assert.deepEqual([...VALID_STATUSES].sort(), [STATUS_FAILURE, STATUS_START, STATUS_SUCCESS].sort())
})

// ---------------------------------------------------------------------------
// fireTelemetry — the fire-and-forget wrapper every call site uses
// ---------------------------------------------------------------------------

test('fireTelemetry: does not block (returns before an async sendFn settles)', () => {
  let resolved = false
  const slowSend = () =>
    new Promise(resolve => {
      setTimeout(() => {
        resolved = true
        resolve({ ok: true })
      }, 50)
    })
  const before = Date.now()
  fireTelemetry(slowSend, { stage: 'x' })
  const elapsedMs = Date.now() - before
  assert.ok(elapsedMs < 20, `fireTelemetry must return immediately, took ${elapsedMs}ms`)
  assert.equal(resolved, false, 'the slow send has not resolved yet — proves fireTelemetry did not await it')
})

test('fireTelemetry: a synchronously-throwing sendFn never escapes', () => {
  assert.doesNotThrow(() => {
    fireTelemetry(
      () => {
        throw new Error('sendFn blew up')
      },
      { stage: 'x' }
    )
  })
})

test('fireTelemetry: a rejecting Promise from sendFn never becomes an unhandled rejection', async () => {
  fireTelemetry(() => Promise.reject(new Error('network down')), { stage: 'x' })
  // Give the microtask queue a turn; if fireTelemetry didn't attach a .catch,
  // node:test would report an unhandledRejection for this file.
  await new Promise(resolve => setImmediate(resolve))
})

test('fireTelemetry: a non-function sendFn (e.g. omitted) is a silent no-op', () => {
  assert.doesNotThrow(() => fireTelemetry(undefined, { stage: 'x' }))
  assert.doesNotThrow(() => fireTelemetry(null, { stage: 'x' }))
})
