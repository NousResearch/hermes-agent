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
  MANAGED_PROVIDER_NAME,
  accessTokenFromLogin,
  accountFromLogin,
  apexWebLoginUrl,
  buildManagedModelConfig,
  decodeJwtClaims,
  defaultModelPath,
  googleStartUrl,
  isLoopbackUrl,
  isManagedEnabled,
  managedModelConfigYaml,
  modelDisabledProvidersYaml,
  seedSkillsBlockYaml,
  MODEL_DISABLED_PROVIDERS,
  SEED_DISABLED_SKILLS,
  parseLoopbackCallback,
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
  // register lives on the auth host alongside login (login-or-register).
  assert.equal(e.registerUrl, 'https://apex-nodes.com/api/v1/auth/register')
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
  // The written/display id derives the collision-free `-APEX` form from the
  // overridden routed model (no explicit display override given).
  assert.equal(e.modelDisplay, 'deepseek-v4-flash-APEX')
  assert.equal(e.loginUrl, 'https://staging.apex-nodes.com/api/v1/auth/login')
  assert.equal(e.provisionKeyUrl, 'https://api.staging.apex-nodes.com/api/v1/desktop/provision-key')
})

test('resolveApexEndpoints modelDisplay precedence: explicit override > derived > default', () => {
  // Prod default
  assert.equal(resolveApexEndpoints({}).modelDisplay, MANAGED_MODEL_DISPLAY)
  // Explicit display override wins even alongside a routed-model override
  assert.equal(
    resolveApexEndpoints({
      APEXNODES_MANAGED_MODEL: 'deepseek-v4-flash',
      APEXNODES_MANAGED_MODEL_DISPLAY: 'custom-label-APEX'
    }).modelDisplay,
    'custom-label-APEX'
  )
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

test('isManagedEnabled is ON by default and accepts common falsy spellings to disable', () => {
  assert.equal(isManagedEnabled({}), true)
  assert.equal(isManagedEnabled({ APEXNODES_MANAGED: '' }), true)
  for (const v of ['0', 'false', 'no', 'off', 'OFF']) {
    assert.equal(isManagedEnabled({ APEXNODES_MANAGED: v }), false, v)
  }
  for (const v of ['1', 'true', 'TRUE', 'yes', 'on']) {
    assert.equal(isManagedEnabled({ APEXNODES_MANAGED: v }), true, v)
  }
})

// --- buildManagedModelConfig + managedModelConfigYaml ---

test('buildManagedModelConfig writes the collision-free DISPLAY model id + registers the relay as a named custom provider', () => {
  const block = buildManagedModelConfig('sk-relaykey123', {})
  assert.deepEqual(block, {
    // The WRITTEN model id is the display name, NOT the raw routed id — the raw
    // `deepseek-v4-pro` collides with the built-in DeepSeek catalog and gets
    // agent init mis-routed to the keyless built-in `deepseek` provider.
    default: 'deepseek-v4-pro-APEX',
    provider: 'custom',
    base_url: 'https://apex-nodes.com/relay/v1',
    api_key: 'sk-relaykey123',
    // The relay is registered as a named custom_providers entry (Hermes-native
    // shape) so the endpoint stays durable across picker switches / resume.
    custom_providers: [
      {
        name: MANAGED_PROVIDER_NAME,
        base_url: 'https://apex-nodes.com/relay/v1',
        api_key: 'sk-relaykey123',
        model: 'deepseek-v4-pro-APEX'
      }
    ]
  })
})

test('buildManagedModelConfig model id is never the catalog-colliding raw routed id', () => {
  // The whole point of the fix: the written id must not be `deepseek-v4-pro`
  // (which is an exact entry in the built-in DeepSeek static catalog).
  const block = buildManagedModelConfig('sk-x', {})
  assert.notEqual(block.default, 'deepseek-v4-pro')
  assert.notEqual(block.custom_providers[0].model, 'deepseek-v4-pro')
  assert.match(block.default, /-APEX$/)
})

test('buildManagedModelConfig custom_providers entry mirrors model.default/base_url/api_key', () => {
  const block = buildManagedModelConfig(
    'sk-x',
    {},
    { baseUrl: 'https://relay.example.com/v1/' }
  )
  // The entry must use the SAME resolved endpoint/model/key as the model block,
  // with the trailing slash stripped, so both anchors point at one place.
  assert.deepEqual(block.custom_providers, [
    {
      name: MANAGED_PROVIDER_NAME,
      base_url: 'https://relay.example.com/v1',
      api_key: 'sk-x',
      model: 'deepseek-v4-pro-APEX'
    }
  ])
  assert.equal(block.base_url, block.custom_providers[0].base_url)
  assert.equal(block.default, block.custom_providers[0].model)
  assert.equal(block.api_key, block.custom_providers[0].api_key)
})

test('buildManagedModelConfig honors a provision-key model only when it is already a non-colliding -APEX id', () => {
  // A raw routed id from the server (e.g. deepseek-v4-pro) must NOT be written
  // verbatim — it would re-seed the collision — so we fall back to the display id.
  const raw = buildManagedModelConfig('sk-x', {}, { model: 'deepseek-v4-pro' })
  assert.equal(raw.default, 'deepseek-v4-pro-APEX')
  assert.equal(raw.custom_providers[0].model, 'deepseek-v4-pro-APEX')
  // An already-branded display id from the server is honored as-is.
  const branded = buildManagedModelConfig('sk-x', {}, { model: 'deepseek-v4-flash-APEX' })
  assert.equal(branded.default, 'deepseek-v4-flash-APEX')
  assert.equal(branded.custom_providers[0].model, 'deepseek-v4-flash-APEX')
})

test('buildManagedModelConfig throws on a missing key (never seed an empty cred)', () => {
  assert.throws(() => buildManagedModelConfig('', {}), /relay key is required/)
  assert.throws(() => buildManagedModelConfig('   ', {}), /relay key is required/)
})

test('buildManagedModelConfig respects env overrides (model id derives the collision-free -APEX display form)', () => {
  const block = buildManagedModelConfig('sk-x', {
    APEXNODES_RELAY_BASE_URL: 'https://staging.apex-nodes.com/relay/v1',
    APEXNODES_MANAGED_MODEL: 'deepseek-v4-flash'
  })
  assert.equal(block.base_url, 'https://staging.apex-nodes.com/relay/v1')
  // A staging routed-model override yields the branded, collision-free written id.
  assert.equal(block.default, 'deepseek-v4-flash-APEX')
})

test('buildManagedModelConfig honors an explicit APEXNODES_MANAGED_MODEL_DISPLAY override', () => {
  const block = buildManagedModelConfig('sk-x', {
    APEXNODES_MANAGED_MODEL: 'deepseek-v4-flash',
    APEXNODES_MANAGED_MODEL_DISPLAY: 'flash-APEX'
  })
  assert.equal(block.default, 'flash-APEX')
  assert.equal(block.custom_providers[0].model, 'flash-APEX')
})

test('buildManagedModelConfig prefers a provision-key base_url override over env defaults', () => {
  const block = buildManagedModelConfig(
    'sk-x',
    {},
    { baseUrl: 'https://relay.example.com/v1/' }
  )
  // overrides win, and the trailing slash is stripped.
  assert.equal(block.base_url, 'https://relay.example.com/v1')
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

test('managedModelConfigYaml emits a valid, quoted model block + custom_providers entry', () => {
  const yaml = managedModelConfigYaml(buildManagedModelConfig('sk-relaykey123', {}))
  assert.match(yaml, /^model:\n/)
  // The collision-free display id is what lands in config.
  assert.match(yaml, /\n {2}default: deepseek-v4-pro-APEX\n/)
  assert.match(yaml, /\n {2}provider: custom\n/)
  // base_url + api_key are double-quoted scalars.
  assert.match(yaml, /\n {2}base_url: "https:\/\/apex-nodes\.com\/relay\/v1"\n/)
  assert.match(yaml, /\n {2}api_key: "sk-relaykey123"\n/)
  // The relay is also registered as a named custom_providers list entry, with
  // the name/base_url/api_key double-quoted and the model id as a bare scalar.
  assert.match(yaml, /\ncustom_providers:\n/)
  assert.match(yaml, /\n {2}- name: "Apex-nodes\.com"\n/)
  assert.match(yaml, /\n {4}base_url: "https:\/\/apex-nodes\.com\/relay\/v1"\n/)
  assert.match(yaml, /\n {4}api_key: "sk-relaykey123"\n/)
  assert.match(yaml, /\n {4}model: deepseek-v4-pro-APEX\n/)
  // Critically, the catalog-colliding raw id must NOT appear as a written model.
  assert.doesNotMatch(yaml, /\n {2}default: deepseek-v4-pro\n/)
})

test('managedModelConfigYaml omits custom_providers when the block has none', () => {
  // Defensive: a block without custom_providers (older shape) still serializes
  // a clean model-only snippet.
  const yaml = managedModelConfigYaml({
    default: 'deepseek-v4-pro',
    provider: 'custom',
    base_url: 'https://apex-nodes.com/relay/v1',
    api_key: 'sk-x'
  })
  assert.match(yaml, /^model:\n/)
  assert.doesNotMatch(yaml, /custom_providers:/)
})

// --- hc-392 China default profile seed helpers ---

test('managedModelConfigYaml nests disabled_providers INSIDE the model block when given', () => {
  const yaml = managedModelConfigYaml(
    buildManagedModelConfig('sk-x', {}),
    { disabledProviders: ['copilot'] }
  )
  // disabled_providers is indented 2 spaces (under model:) and appears before
  // the top-level custom_providers: key — never as a second top-level model:.
  assert.match(yaml, /\n {2}disabled_providers:\n {4}- copilot\n/)
  assert.equal((yaml.match(/^model:\n/gm) || []).length, 1)
  const dpIdx = yaml.indexOf('disabled_providers:')
  const cpIdx = yaml.indexOf('custom_providers:')
  assert.ok(dpIdx !== -1 && cpIdx !== -1 && dpIdx < cpIdx)
})

test('managedModelConfigYaml omits disabled_providers when not requested (back-compat)', () => {
  const yaml = managedModelConfigYaml(buildManagedModelConfig('sk-x', {}))
  assert.doesNotMatch(yaml, /disabled_providers/)
})

test('modelDisabledProvidersYaml defaults to [copilot], indented under model:', () => {
  assert.equal(modelDisabledProvidersYaml(), '  disabled_providers:\n    - copilot\n')
  assert.deepEqual(MODEL_DISABLED_PROVIDERS, ['copilot'])
  // Empty/whitespace lists render nothing (so the seed stays clean).
  assert.equal(modelDisabledProvidersYaml([]), '')
  assert.equal(modelDisabledProvidersYaml(['  ']), '')
})

test('seedSkillsBlockYaml emits a top-level skills.disabled block with all 49 names', () => {
  const yaml = seedSkillsBlockYaml()
  assert.match(yaml, /^skills:\n {2}disabled:\n/m)
  assert.equal(SEED_DISABLED_SKILLS.length, 49)
  assert.equal(new Set(SEED_DISABLED_SKILLS).size, 49) // no dupes
  // Every name is rendered as a 4-space-indented list item.
  for (const name of SEED_DISABLED_SKILLS) {
    assert.ok(yaml.includes(`\n    - ${name}\n`), `missing seeded skill: ${name}`)
  }
  // The four frontmatter-name (≠ folder) skills must be present by their
  // frontmatter name, or the runtime toggle won't match them.
  for (const n of ['serving-llms-vllm', 'evaluating-llms-harness', 'segment-anything-model', 'audiocraft-audio-generation']) {
    assert.ok(SEED_DISABLED_SKILLS.includes(n), `name-mismatch skill not seeded: ${n}`)
  }
  // The 17 hard-cut skills are all in the disabled set.
  for (const n of ['google-workspace', 'xurl', 'youtube-content', 'polymarket', 'teams-meeting-pipeline', 'claude-code', 'codex', 'opencode', 'notion', 'airtable', 'gif-search', 'arxiv', 'github-auth', 'github-code-review', 'github-issues', 'github-pr-workflow', 'github-repo-management']) {
    assert.ok(SEED_DISABLED_SKILLS.includes(n), `cut skill not seeded: ${n}`)
  }
  assert.equal(seedSkillsBlockYaml([]), '')
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

// --- googleStartUrl / apexWebLoginUrl (browser-login start URLs) ---

test('googleStartUrl points at the API host with loopback redirect_uri + state', () => {
  const url = googleStartUrl('http://127.0.0.1:51234/cb', 'st-abc', {})
  const u = new URL(url)
  assert.equal(u.origin, 'https://api.apex-nodes.com')
  assert.equal(u.pathname, '/api/v1/auth/google/start')
  assert.equal(u.searchParams.get('redirect_uri'), 'http://127.0.0.1:51234/cb')
  assert.equal(u.searchParams.get('state'), 'st-abc')
})

test('googleStartUrl honors APEXNODES_API_BASE override', () => {
  const url = googleStartUrl('http://127.0.0.1:1/cb', 's', {
    APEXNODES_API_BASE: 'https://api.staging.apex-nodes.com/'
  })
  assert.equal(new URL(url).origin, 'https://api.staging.apex-nodes.com')
})

test('apexWebLoginUrl opens the zh web login page with desktop_cb + state', () => {
  const url = apexWebLoginUrl('http://127.0.0.1:51234/cb', 'st-xyz', {})
  const u = new URL(url)
  assert.equal(u.origin, 'https://apex-nodes.com')
  assert.equal(u.pathname, '/zh/login')
  assert.equal(u.searchParams.get('desktop_cb'), 'http://127.0.0.1:51234/cb')
  assert.equal(u.searchParams.get('state'), 'st-xyz')
})

test('apexWebLoginUrl honors APEXNODES_AUTH_BASE override', () => {
  const url = apexWebLoginUrl('http://127.0.0.1:1/cb', 's', {
    APEXNODES_AUTH_BASE: 'https://staging.apex-nodes.com'
  })
  assert.equal(new URL(url).origin, 'https://staging.apex-nodes.com')
})

// --- parseLoopbackCallback (CSRF-validated browser redirect) ---

test('parseLoopbackCallback accepts a /cb hit with matching state + token', () => {
  const out = parseLoopbackCallback('/cb?token=jwt.123&state=s1', 's1')
  assert.deepEqual(out, { ok: true, token: 'jwt.123' })
})

test('parseLoopbackCallback trims the token', () => {
  const out = parseLoopbackCallback('/cb?token=%20jwt.trim%20&state=s1', 's1')
  assert.deepEqual(out, { ok: true, token: 'jwt.trim' })
})

test('parseLoopbackCallback rejects a state mismatch (CSRF)', () => {
  const out = parseLoopbackCallback('/cb?token=jwt.123&state=evil', 's1')
  assert.equal(out.ok, false)
  assert.equal(out.reason, 'state_mismatch')
  assert.equal(out.isCallback, true)
})

test('parseLoopbackCallback rejects an empty expected state', () => {
  const out = parseLoopbackCallback('/cb?token=jwt.123&state=', '')
  assert.equal(out.ok, false)
  assert.equal(out.reason, 'state_mismatch')
})

test('parseLoopbackCallback rejects a missing token even when state matches', () => {
  const out = parseLoopbackCallback('/cb?state=s1', 's1')
  assert.equal(out.ok, false)
  assert.equal(out.reason, 'missing_token')
})

test('parseLoopbackCallback surfaces an explicit ?error= from the backend', () => {
  const out = parseLoopbackCallback('/cb?error=access_denied&state=s1', 's1')
  assert.equal(out.ok, false)
  assert.equal(out.reason, 'access_denied')
  assert.equal(out.isCallback, true)
})

test('parseLoopbackCallback ignores non-/cb requests (e.g. favicon)', () => {
  const out = parseLoopbackCallback('/favicon.ico', 's1')
  assert.equal(out.ok, false)
  assert.equal(out.isCallback, false)
  assert.equal(out.reason, 'not_callback')
})

// --- isLoopbackUrl ---

test('isLoopbackUrl is true only for http loopback hosts', () => {
  assert.equal(isLoopbackUrl('http://127.0.0.1:1234/cb'), true)
  assert.equal(isLoopbackUrl('http://localhost:1234/cb'), true)
  assert.equal(isLoopbackUrl('http://[::1]:1234/cb'), true)
  // https or non-loopback hosts are rejected.
  assert.equal(isLoopbackUrl('https://127.0.0.1/cb'), false)
  assert.equal(isLoopbackUrl('http://example.com/cb'), false)
  assert.equal(isLoopbackUrl('http://169.254.169.254/cb'), false)
  assert.equal(isLoopbackUrl('not a url'), false)
})

// Build a JWT-shaped string (header.payload.sig) from a claims object; only the
// payload segment is read by decodeJwtClaims (signature never verified here).
function makeJwt(claims) {
  const payload = Buffer.from(JSON.stringify(claims))
    .toString('base64')
    .replace(/=+$/, '')
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
  return `header.${payload}.sig`
}

test('decodeJwtClaims reads the payload segment without verifying the signature', () => {
  const token = makeJwt({ email: 'jane@apex-nodes.com', plan: 'pro', sub: 'u_123' })
  assert.deepEqual(decodeJwtClaims(token), { email: 'jane@apex-nodes.com', plan: 'pro', sub: 'u_123' })
})

test('decodeJwtClaims returns {} for malformed / empty input (never throws)', () => {
  assert.deepEqual(decodeJwtClaims(''), {})
  assert.deepEqual(decodeJwtClaims('not-a-jwt'), {})
  assert.deepEqual(decodeJwtClaims('a.b'), {}) // "b" isn't valid base64 JSON
  assert.deepEqual(decodeJwtClaims(null), {})
  assert.deepEqual(decodeJwtClaims(undefined), {})
})

test('accountFromLogin prefers the response body, falls back to JWT claims', () => {
  // Body wins when present.
  const token = makeJwt({ email: 'claims@apex-nodes.com', plan: 'free' })
  assert.deepEqual(accountFromLogin({ email: 'body@apex-nodes.com', plan: 'pro', name: 'Jane' }, token), {
    email: 'body@apex-nodes.com',
    name: 'Jane',
    plan: 'pro'
  })
})

test('accountFromLogin falls back to JWT email/plan when the body omits them', () => {
  const token = makeJwt({ email: 'claims@apex-nodes.com', tier: 'pro' })
  assert.deepEqual(accountFromLogin({}, token), { email: 'claims@apex-nodes.com', name: '', plan: 'pro' })
})

test('accountFromLogin uses an @-shaped JWT sub as an email fallback', () => {
  const token = makeJwt({ sub: 'sub@apex-nodes.com' })
  assert.equal(accountFromLogin({}, token).email, 'sub@apex-nodes.com')
  // A non-email sub (opaque id) is NOT used as an email.
  assert.equal(accountFromLogin({}, makeJwt({ sub: 'u_123' })).email, '')
})

test('accountFromLogin returns empty strings when nothing is available', () => {
  assert.deepEqual(accountFromLogin(null, ''), { email: '', name: '', plan: '' })
  assert.deepEqual(accountFromLogin(undefined, undefined), { email: '', name: '', plan: '' })
})
