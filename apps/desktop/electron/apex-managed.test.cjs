/**
 * Tests for electron/apex-managed.cjs.
 *
 * Run with: node --test electron/apex-managed.test.cjs
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * These are the pure helpers behind the ApexNodes managed-LLM default path:
 * endpoint resolution (+ env overrides), the managed config.yaml block builder
 * and its YAML serialization, the enable gate, and the response parsers.
 */

const test = require('node:test')
const assert = require('node:assert/strict')

const {
  DEFAULT_AUTH_BASE,
  DEFAULT_API_BASE,
  DEFAULT_RELAY_BASE_URL,
  DEFAULT_MANAGED_MODEL,
  MANAGED_MODEL_DISPLAY,
  MANAGED_PROVIDER,
  accessTokenFromLogin,
  buildManagedModelConfig,
  defaultModelPath,
  isManagedEnabled,
  managedModelConfigYaml,
  parseProvisionResponse,
  relayKeyFromResponse,
  resolveApexEndpoints
} = require('./apex-managed.cjs')

// --- resolveApexEndpoints ---

test('resolveApexEndpoints returns prod defaults with empty env', () => {
  const e = resolveApexEndpoints({})
  assert.equal(e.authBase, DEFAULT_AUTH_BASE)
  assert.equal(e.apiBase, DEFAULT_API_BASE)
  assert.equal(e.relayBaseUrl, DEFAULT_RELAY_BASE_URL)
  assert.equal(e.model, DEFAULT_MANAGED_MODEL)
  assert.equal(e.modelDisplay, MANAGED_MODEL_DISPLAY)
  assert.equal(e.provider, MANAGED_PROVIDER)
  assert.equal(e.loginUrl, 'https://apex-nodes.com/api/v1/auth/login')
  // provision-key is on the API host per the P0 contract.
  assert.equal(e.provisionKeyUrl, 'https://api.apex-nodes.com/api/v1/desktop/provision-key')
})

test('resolveApexEndpoints honors env overrides and strips trailing slashes', () => {
  const e = resolveApexEndpoints({
    APEXNODES_AUTH_BASE: 'https://staging.apex-nodes.com/',
    APEXNODES_API_BASE: 'https://api.staging.apex-nodes.com/',
    APEXNODES_RELAY_BASE_URL: 'https://staging.apex-nodes.com/relay/v1/',
    APEXNODES_MANAGED_MODEL: 'deepseek-v4-flash'
  })
  assert.equal(e.authBase, 'https://staging.apex-nodes.com')
  assert.equal(e.apiBase, 'https://api.staging.apex-nodes.com')
  assert.equal(e.relayBaseUrl, 'https://staging.apex-nodes.com/relay/v1')
  assert.equal(e.model, 'deepseek-v4-flash')
  assert.equal(e.loginUrl, 'https://staging.apex-nodes.com/api/v1/auth/login')
  assert.equal(e.provisionKeyUrl, 'https://api.staging.apex-nodes.com/api/v1/desktop/provision-key')
})

test('relay base_url + /chat/completions reaches the relay route after nginx strips /relay', () => {
  // The runtime appends /chat/completions to model.base_url; nginx maps
  // /relay/ -> :7000/ (prefix stripped). So the relay sees /v1/chat/completions.
  const { relayBaseUrl } = resolveApexEndpoints({})
  const requestUrl = `${relayBaseUrl}/chat/completions`
  assert.equal(requestUrl, 'https://apex-nodes.com/relay/v1/chat/completions')
  const afterNginx = requestUrl.replace('https://apex-nodes.com/relay', '')
  assert.equal(afterNginx, '/v1/chat/completions')
})

// --- isManagedEnabled ---

test('isManagedEnabled is OFF by default and accepts common truthy spellings', () => {
  assert.equal(isManagedEnabled({}), false)
  assert.equal(isManagedEnabled({ APEXNODES_MANAGED: '0' }), false)
  assert.equal(isManagedEnabled({ APEXNODES_MANAGED: 'false' }), false)
  assert.equal(isManagedEnabled({ APEXNODES_MANAGED: '' }), false)
  for (const v of ['1', 'true', 'TRUE', 'yes', 'on']) {
    assert.equal(isManagedEnabled({ APEXNODES_MANAGED: v }), true, v)
  }
})

// --- buildManagedModelConfig + managedModelConfigYaml ---

test('buildManagedModelConfig wires provider/base_url/api_key/default', () => {
  const block = buildManagedModelConfig('sk-relaykey123', {})
  assert.deepEqual(block, {
    default: 'deepseek-v4-pro',
    provider: 'custom',
    base_url: 'https://apex-nodes.com/relay/v1',
    api_key: 'sk-relaykey123'
  })
})

test('buildManagedModelConfig throws on a missing key (never seed an empty cred)', () => {
  assert.throws(() => buildManagedModelConfig('', {}), /relay key is required/)
  assert.throws(() => buildManagedModelConfig('   ', {}), /relay key is required/)
})

test('buildManagedModelConfig respects env overrides', () => {
  const block = buildManagedModelConfig('sk-x', {
    APEXNODES_RELAY_BASE_URL: 'https://staging.apex-nodes.com/relay/v1',
    APEXNODES_MANAGED_MODEL: 'deepseek-v4-flash'
  })
  assert.equal(block.base_url, 'https://staging.apex-nodes.com/relay/v1')
  assert.equal(block.default, 'deepseek-v4-flash')
})

test('buildManagedModelConfig prefers provision-key overrides over env defaults', () => {
  const block = buildManagedModelConfig(
    'sk-x',
    {},
    { baseUrl: 'https://relay.example.com/v1/', model: 'deepseek-v4-special' }
  )
  // overrides win, and the trailing slash is stripped.
  assert.equal(block.base_url, 'https://relay.example.com/v1')
  assert.equal(block.default, 'deepseek-v4-special')
  assert.equal(block.provider, 'custom')
})

// --- parseProvisionResponse ---

test('parseProvisionResponse extracts key + base_url + model from the contract shape', () => {
  const out = parseProvisionResponse(
    { api_key: 'sk-relay', base_url: 'https://apex-nodes.com/relay/v1', model: 'deepseek-v4-pro' },
    {}
  )
  assert.deepEqual(out, {
    apiKey: 'sk-relay',
    baseUrl: 'https://apex-nodes.com/relay/v1',
    model: 'deepseek-v4-pro'
  })
})

test('parseProvisionResponse falls back to env defaults when server omits base_url/model', () => {
  const out = parseProvisionResponse({ api_key: 'sk-relay' }, {})
  assert.equal(out.apiKey, 'sk-relay')
  assert.equal(out.baseUrl, DEFAULT_RELAY_BASE_URL)
  assert.equal(out.model, DEFAULT_MANAGED_MODEL)
})

test('parseProvisionResponse returns null without a key (fall back to BYOK)', () => {
  assert.equal(parseProvisionResponse({ base_url: 'x', model: 'y' }, {}), null)
  assert.equal(parseProvisionResponse({}, {}), null)
  assert.equal(parseProvisionResponse(null, {}), null)
})

test('managedModelConfigYaml emits a valid, quoted model block', () => {
  const yaml = managedModelConfigYaml(buildManagedModelConfig('sk-relaykey123', {}))
  assert.match(yaml, /^model:\n/)
  assert.match(yaml, /\n {2}default: deepseek-v4-pro\n/)
  assert.match(yaml, /\n {2}provider: custom\n/)
  // base_url + api_key are double-quoted scalars.
  assert.match(yaml, /\n {2}base_url: "https:\/\/apex-nodes\.com\/relay\/v1"\n/)
  assert.match(yaml, /\n {2}api_key: "sk-relaykey123"\n/)
})

// --- defaultModelPath ---

test('defaultModelPath returns managed only when enabled AND a key is present', () => {
  assert.equal(defaultModelPath({ enabled: true, key: 'sk-x' }), 'managed')
  assert.equal(defaultModelPath({ enabled: true, key: null }), 'byok')
  assert.equal(defaultModelPath({ enabled: true, key: '   ' }), 'byok')
  assert.equal(defaultModelPath({ enabled: false, key: 'sk-x' }), 'byok')
  assert.equal(defaultModelPath({}), 'byok')
})

// --- relayKeyFromResponse ---

test('relayKeyFromResponse tolerates the likely backend shapes', () => {
  assert.equal(relayKeyFromResponse({ relay_key: 'sk-a' }), 'sk-a')
  assert.equal(relayKeyFromResponse({ api_key: 'sk-b' }), 'sk-b')
  assert.equal(relayKeyFromResponse({ key: 'sk-c' }), 'sk-c')
  assert.equal(relayKeyFromResponse({ item: { key: 'sk-d' } }), 'sk-d')
  assert.equal(relayKeyFromResponse({ item: { api_key: 'sk-e' } }), 'sk-e')
})

test('relayKeyFromResponse returns null for empty/garbage (fall back to BYOK)', () => {
  assert.equal(relayKeyFromResponse(null), null)
  assert.equal(relayKeyFromResponse({}), null)
  assert.equal(relayKeyFromResponse({ key: '' }), null)
  assert.equal(relayKeyFromResponse({ key: '   ' }), null)
  assert.equal(relayKeyFromResponse('sk-not-an-object'), null)
})

// --- accessTokenFromLogin ---

test('accessTokenFromLogin extracts the JWT or null', () => {
  assert.equal(accessTokenFromLogin({ access_token: 'jwt.abc', token_type: 'bearer' }), 'jwt.abc')
  assert.equal(accessTokenFromLogin({ access_token: '  jwt.trim  ' }), 'jwt.trim')
  assert.equal(accessTokenFromLogin({}), null)
  assert.equal(accessTokenFromLogin({ access_token: '' }), null)
  assert.equal(accessTokenFromLogin(null), null)
})
