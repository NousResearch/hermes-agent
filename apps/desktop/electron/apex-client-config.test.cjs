/**
 * Tests for electron/apex-client-config.cjs.
 *
 * Run with: node --test electron/apex-client-config.test.cjs
 *
 * These are the pure helpers behind the platform client-config sync: response
 * validation/normalization, the apply predicate, the persisted-state
 * normalizer, and the fail-soft fetch wrapper (exercised against stub
 * fetchJson implementations — no network).
 */

const test = require('node:test')
const assert = require('node:assert/strict')

const {
  CLIENT_CONFIG_PATH,
  clientConfigUrl,
  fetchClientConfig,
  normalizeStoredClientConfig,
  normalizeVersion,
  parseClientConfigResponse,
  shouldApply
} = require('./apex-client-config.cjs')

// --- clientConfigUrl ---

test('clientConfigUrl builds the public endpoint and strips trailing slashes', () => {
  assert.equal(
    clientConfigUrl('https://api.apex-nodes.com'),
    `https://api.apex-nodes.com${CLIENT_CONFIG_PATH}`
  )
  assert.equal(
    clientConfigUrl('https://api.staging.apex-nodes.com///'),
    `https://api.staging.apex-nodes.com${CLIENT_CONFIG_PATH}`
  )
})

test('clientConfigUrl advertises known_version only when it is a positive int', () => {
  assert.equal(
    clientConfigUrl('https://api.apex-nodes.com', 7),
    'https://api.apex-nodes.com/api/v1/desktop/client-config?known_version=7'
  )
  // 0 / garbage → no query (a first fetch must get the full payload).
  assert.equal(clientConfigUrl('https://api.apex-nodes.com', 0), `https://api.apex-nodes.com${CLIENT_CONFIG_PATH}`)
  assert.equal(clientConfigUrl('https://api.apex-nodes.com', -3), `https://api.apex-nodes.com${CLIENT_CONFIG_PATH}`)
  assert.equal(
    clientConfigUrl('https://api.apex-nodes.com', 'nope'),
    `https://api.apex-nodes.com${CLIENT_CONFIG_PATH}`
  )
})

// --- normalizeVersion ---

test('normalizeVersion accepts positive ints (number or digit string), rejects the rest', () => {
  assert.equal(normalizeVersion(3), 3)
  assert.equal(normalizeVersion('12'), 12)
  assert.equal(normalizeVersion(' 4 '), 4)
  assert.equal(normalizeVersion(0), 0)
  assert.equal(normalizeVersion(-1), 0)
  assert.equal(normalizeVersion(1.5), 0)
  assert.equal(normalizeVersion('1.5'), 0)
  assert.equal(normalizeVersion('v2'), 0)
  assert.equal(normalizeVersion(null), 0)
  assert.equal(normalizeVersion(undefined), 0)
  assert.equal(normalizeVersion({}), 0)
  assert.equal(normalizeVersion(true), 0)
})

// --- parseClientConfigResponse ---

test('parseClientConfigResponse normalizes a good full response', () => {
  const parsed = parseClientConfigResponse({
    version: 3,
    payload: { config_yaml: { 'display.show_reasoning': true } },
    updated_at: '2026-07-03T00:00:00Z'
  })
  assert.deepEqual(parsed, {
    version: 3,
    payload: { config_yaml: { 'display.show_reasoning': true } },
    unchanged: false,
    updatedAt: '2026-07-03T00:00:00Z'
  })
})

test('parseClientConfigResponse tolerates a numeric-string version and missing updated_at', () => {
  const parsed = parseClientConfigResponse({ version: '5', payload: {} })
  assert.deepEqual(parsed, { version: 5, payload: {}, unchanged: false, updatedAt: null })
})

test('parseClientConfigResponse preserves unknown fields inside payload (forward compat)', () => {
  const parsed = parseClientConfigResponse({
    version: 2,
    payload: { config_yaml: { 'a.b': 1 }, future_field: { anything: true } },
    some_new_top_level: 'ignored'
  })
  assert.ok(parsed)
  assert.equal(parsed.version, 2)
  // The payload rides along verbatim; consumers only read the keys they know.
  assert.deepEqual(parsed.payload.future_field, { anything: true })
  assert.equal('some_new_top_level' in parsed, false)
})

test('parseClientConfigResponse handles the unchanged short-circuit shape', () => {
  const parsed = parseClientConfigResponse({ version: 9, unchanged: true })
  assert.deepEqual(parsed, { version: 9, payload: null, unchanged: true, updatedAt: null })
})

test('parseClientConfigResponse returns null on garbage', () => {
  assert.equal(parseClientConfigResponse(null), null)
  assert.equal(parseClientConfigResponse(undefined), null)
  assert.equal(parseClientConfigResponse('a string'), null)
  assert.equal(parseClientConfigResponse(42), null)
  assert.equal(parseClientConfigResponse([]), null)
  assert.equal(parseClientConfigResponse({}), null)
  // Bad versions.
  assert.equal(parseClientConfigResponse({ version: 0, payload: {} }), null)
  assert.equal(parseClientConfigResponse({ version: -2, payload: {} }), null)
  assert.equal(parseClientConfigResponse({ version: 1.5, payload: {} }), null)
  assert.equal(parseClientConfigResponse({ version: 'x', payload: {} }), null)
  // Bad payloads (full shape requires an object payload).
  assert.equal(parseClientConfigResponse({ version: 1 }), null)
  assert.equal(parseClientConfigResponse({ version: 1, payload: null }), null)
  assert.equal(parseClientConfigResponse({ version: 1, payload: [] }), null)
  assert.equal(parseClientConfigResponse({ version: 1, payload: 'yaml' }), null)
  // unchanged must be literally true to count as the short-circuit shape.
  assert.equal(parseClientConfigResponse({ version: 1, unchanged: 'true' }), null)
})

// --- shouldApply ---

test('shouldApply matrix', () => {
  // Strictly newer → apply.
  assert.equal(shouldApply(2, 1), true)
  assert.equal(shouldApply(10, 9), true)
  // First ever config (nothing applied yet) → apply.
  assert.equal(shouldApply(1, 0), true)
  assert.equal(shouldApply(1, null), true)
  assert.equal(shouldApply(1, undefined), true)
  assert.equal(shouldApply(1, 'garbage'), true)
  // Same version → no-op.
  assert.equal(shouldApply(1, 1), false)
  assert.equal(shouldApply(7, 7), false)
  // Version regress → never re-apply an older payload.
  assert.equal(shouldApply(1, 2), false)
  assert.equal(shouldApply(3, 10), false)
  // Garbage fetched version → never apply.
  assert.equal(shouldApply(0, 0), false)
  assert.equal(shouldApply(-1, 0), false)
  assert.equal(shouldApply(1.5, 0), false)
  assert.equal(shouldApply(null, 0), false)
  assert.equal(shouldApply('x', 0), false)
  // Digit strings normalize on both sides.
  assert.equal(shouldApply('3', '2'), true)
  assert.equal(shouldApply('2', '2'), false)
})

// --- normalizeStoredClientConfig ---

test('normalizeStoredClientConfig round-trips a good persisted state', () => {
  const state = {
    version: 4,
    payload: { config_yaml: { 'display.show_reasoning': true } },
    fetchedAt: 1_750_000_000_000,
    appliedVersion: 3
  }
  assert.deepEqual(normalizeStoredClientConfig(state), state)
})

test('normalizeStoredClientConfig degrades garbage to the empty state', () => {
  const empty = { version: 0, payload: null, fetchedAt: null, appliedVersion: 0 }
  assert.deepEqual(normalizeStoredClientConfig(null), empty)
  assert.deepEqual(normalizeStoredClientConfig('corrupt'), empty)
  assert.deepEqual(normalizeStoredClientConfig([]), empty)
  assert.deepEqual(normalizeStoredClientConfig({}), empty)
  // A version without its payload (or vice versa) is unusable.
  assert.deepEqual(normalizeStoredClientConfig({ version: 2 }), empty)
  assert.deepEqual(normalizeStoredClientConfig({ payload: {} }), empty)
  assert.deepEqual(normalizeStoredClientConfig({ version: 2, payload: [] }), empty)
})

test('normalizeStoredClientConfig cleans partial fields on a usable state', () => {
  const normalized = normalizeStoredClientConfig({
    version: '6',
    payload: { config_yaml: {} },
    fetchedAt: 'not-a-number',
    appliedVersion: -9
  })
  assert.deepEqual(normalized, {
    version: 6,
    payload: { config_yaml: {} },
    fetchedAt: null,
    appliedVersion: 0
  })
})

// --- fetchClientConfig (stubbed fetchJson; never throws) ---

test('fetchClientConfig resolves the parsed config and passes known_version', async () => {
  const calls = []
  const fetchJson = async (url, options) => {
    calls.push({ url, options })
    return { version: 2, payload: { config_yaml: { 'a.b': 'x' } } }
  }
  const result = await fetchClientConfig({
    apiBase: 'https://api.apex-nodes.com/',
    fetchJson,
    knownVersion: 1,
    timeoutMs: 1234
  })
  assert.deepEqual(result, {
    version: 2,
    payload: { config_yaml: { 'a.b': 'x' } },
    unchanged: false,
    updatedAt: null
  })
  assert.equal(calls.length, 1)
  assert.equal(calls[0].url, 'https://api.apex-nodes.com/api/v1/desktop/client-config?known_version=1')
  assert.equal(calls[0].options.timeoutMs, 1234)
})

test('fetchClientConfig returns null on a rejected fetch (404 / offline) and logs', async () => {
  const logs = []
  const result = await fetchClientConfig({
    apiBase: 'https://api.apex-nodes.com',
    fetchJson: async () => {
      throw new Error('404: no active config')
    },
    log: msg => logs.push(msg)
  })
  assert.equal(result, null)
  assert.ok(logs.some(line => line.includes('404')))
})

test('fetchClientConfig returns null on a malformed body', async () => {
  const result = await fetchClientConfig({
    apiBase: 'https://api.apex-nodes.com',
    fetchJson: async () => ({ nonsense: true })
  })
  assert.equal(result, null)
})

test('fetchClientConfig surfaces the unchanged short-circuit', async () => {
  const result = await fetchClientConfig({
    apiBase: 'https://api.apex-nodes.com',
    fetchJson: async () => ({ version: 5, unchanged: true })
  })
  assert.deepEqual(result, { version: 5, payload: null, unchanged: true, updatedAt: null })
})

test('fetchClientConfig returns null without an apiBase or fetchJson', async () => {
  assert.equal(await fetchClientConfig({ apiBase: '', fetchJson: async () => ({}) }), null)
  assert.equal(await fetchClientConfig({ apiBase: 'https://api.apex-nodes.com' }), null)
})
