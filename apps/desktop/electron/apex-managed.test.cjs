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
  ensurePluginsEnabledYaml,
  ensureSkillsDisabledYaml,
  modelDisabledProvidersYaml,
  seedSkillsBlockYaml,
  seedPluginsBlockYaml,
  MODEL_DISABLED_PROVIDERS,
  SEED_DISABLED_SKILLS,
  MANAGED_PLUGIN_NAMES,
  parseLoopbackCallback,
  parseProvisionResponse,
  relayKeyFromResponse,
  resolveApexEndpoints,
  syncCustomProviderKeyYaml
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
    model: 'deepseek-v4-pro',
    email: '',
    name: '',
    plan: ''
  })
})

test('parseProvisionResponse captures the display-only identity (email/name/plan)', () => {
  const out = parseProvisionResponse(
    { api_key: 'sk-relay', email: '  user@apex-nodes.com  ', name: 'Kael', plan: 'pro' },
    {}
  )
  assert.equal(out.email, 'user@apex-nodes.com')
  assert.equal(out.name, 'Kael')
  assert.equal(out.plan, 'pro')
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

test('seedSkillsBlockYaml emits a top-level skills.disabled block with all 50 v0.18 names', () => {
  const yaml = seedSkillsBlockYaml()
  assert.match(yaml, /^skills:\n {2}disabled:\n/m)
  // hc-406 v0.18 全集重分级: 50 disabled bundled skills (was 49 @ v0.17;
  // +huggingface-hub/maps/plan, −dead kanban-orchestrator/kanban-worker).
  assert.equal(SEED_DISABLED_SKILLS.length, 50)
  assert.equal(new Set(SEED_DISABLED_SKILLS).size, 50) // no dupes
  // Every name is rendered as a 4-space-indented list item.
  for (const name of SEED_DISABLED_SKILLS) {
    assert.ok(yaml.includes(`\n    - ${name}\n`), `missing seeded skill: ${name}`)
  }
  // The four frontmatter-name (≠ folder) skills must be present by their
  // frontmatter name, or the runtime toggle won't match them.
  for (const n of ['serving-llms-vllm', 'evaluating-llms-harness', 'segment-anything-model', 'audiocraft-audio-generation']) {
    assert.ok(SEED_DISABLED_SKILLS.includes(n), `name-mismatch skill not seeded: ${n}`)
  }
  // The hard-cut (D) walled/geo/competitor skills are all in the disabled set.
  for (const n of ['google-workspace', 'xurl', 'youtube-content', 'polymarket', 'teams-meeting-pipeline', 'claude-code', 'codex', 'gif-search']) {
    assert.ok(SEED_DISABLED_SKILLS.includes(n), `cut skill not seeded: ${n}`)
  }
  // hc-406: the three newly-bundled skills graded OFF must be present.
  for (const n of ['huggingface-hub', 'maps', 'plan']) {
    assert.ok(SEED_DISABLED_SKILLS.includes(n), `new-v0.18 disabled skill not seeded: ${n}`)
  }
  // hc-406: the dead kanban orphans (no matching bundled skill anymore) are gone.
  for (const n of ['kanban-orchestrator', 'kanban-worker']) {
    assert.ok(!SEED_DISABLED_SKILLS.includes(n), `dead orphan should be dropped: ${n}`)
  }
  // hc-406: newly-bundled A/B skills we KEEP ACTIVE must NOT be disabled.
  for (const n of ['computer-use', 'powerpoint', 'obsidian', 'ocr-and-documents', 'baoyu-infographic', 'yuanbao', 'apple-notes', 'imessage', 'llm-wiki', 'blogwatcher']) {
    assert.ok(!SEED_DISABLED_SKILLS.includes(n), `A/B skill must stay active: ${n}`)
  }
  assert.equal(seedSkillsBlockYaml([]), '')
})

// --- plugins seed + guard (desktop-plugins-seed) ---
// The runtime's standalone plugin loader is opt-in (only `plugins.enabled`
// names load), so the desktop seed must carry the block and the config
// watchdog must be able to heal a config.yaml that lost (or predates) it.

test('MANAGED_PLUGIN_NAMES carries apex-overlay + the three platform tool plugins', () => {
  assert.deepEqual(MANAGED_PLUGIN_NAMES, [
    'apex-overlay',
    'apexnodes-douyin-tools',
    'apexnodes-social-tools',
    'apexnodes-video-tools'
  ])
})

test('seedPluginsBlockYaml emits a top-level plugins.enabled block with all managed plugins', () => {
  const yaml = seedPluginsBlockYaml()
  assert.match(yaml, /^plugins:\n {2}enabled:\n/m)
  // Every name rendered exactly once, as a 4-space-indented list item (the
  // cli-config.yaml.example shape).
  for (const name of MANAGED_PLUGIN_NAMES) {
    assert.ok(yaml.includes(`\n    - ${name}\n`), `missing seeded plugin: ${name}`)
  }
  assert.equal(yaml.split('\n').filter(l => /^ {4}- /.test(l)).length, MANAGED_PLUGIN_NAMES.length)
  assert.equal(seedPluginsBlockYaml([]), '')
})

test('a fresh desktop seed already satisfies the plugins guard (fresh install = no-op)', () => {
  // Same composition order as seedDefaultModelConfig's BYOK path.
  const seed =
    'model:\n  default: deepseek-v4-pro\n  provider: deepseek\n' +
    modelDisabledProvidersYaml() +
    seedSkillsBlockYaml() +
    seedPluginsBlockYaml()
  const r = ensurePluginsEnabledYaml(seed)
  assert.equal(r.changed, false)
  assert.equal(r.next, seed)
  assert.deepEqual(r.added, [])
})

test('ensurePluginsEnabledYaml appends the full block when plugins: is missing (pre-seed install)', () => {
  const raw = 'model:\n  default: deepseek-v4-pro\nskills:\n  disabled:\n    - notion\n'
  const r = ensurePluginsEnabledYaml(raw)
  assert.equal(r.changed, true)
  assert.deepEqual(r.added, MANAGED_PLUGIN_NAMES)
  // Append-only: the original text is untouched, one plugins: block added.
  assert.ok(r.next.startsWith(raw))
  assert.equal((r.next.match(/^plugins:/gm) || []).length, 1)
  for (const name of MANAGED_PLUGIN_NAMES) {
    assert.ok(r.next.includes(`\n    - ${name}\n`), `missing healed plugin: ${name}`)
  }
})

test('ensurePluginsEnabledYaml unions missing managed names and preserves user entries', () => {
  const raw =
    'plugins:\n' +
    '  enabled:\n' +
    '    - my-own-plugin\n' +
    '    - apex-overlay\n' +
    'model:\n' +
    '  default: x\n'
  const r = ensurePluginsEnabledYaml(raw)
  assert.equal(r.changed, true)
  assert.deepEqual(r.added, ['apexnodes-douyin-tools', 'apexnodes-social-tools', 'apexnodes-video-tools'])
  // User entry + order preserved; apex-overlay NOT duplicated.
  assert.match(r.next, /enabled:\n {4}- my-own-plugin\n {4}- apex-overlay\n {4}- apexnodes-douyin-tools/)
  assert.equal((r.next.match(/- apex-overlay/g) || []).length, 1)
  // Inserted INSIDE the plugins block (before the next top-level key).
  assert.ok(r.next.indexOf('- apexnodes-video-tools') < r.next.indexOf('model:'))
})

test('ensurePluginsEnabledYaml is idempotent (re-run is a no-op, no duplicates)', () => {
  const first = ensurePluginsEnabledYaml('display:\n  language: zh\n')
  assert.equal(first.changed, true)
  const second = ensurePluginsEnabledYaml(first.next)
  assert.equal(second.changed, false)
  assert.equal(second.next, first.next)
  for (const name of MANAGED_PLUGIN_NAMES) {
    assert.equal((first.next.match(new RegExp(`- ${name}`, 'g')) || []).length, 1, `duplicated: ${name}`)
  }
})

test('ensurePluginsEnabledYaml fills an empty enabled list (bare key / [] / null)', () => {
  for (const shape of ['  enabled:\n', '  enabled: []\n', '  enabled: null\n']) {
    const r = ensurePluginsEnabledYaml(`plugins:\n${shape}model:\n  default: x\n`)
    assert.equal(r.changed, true, `shape: ${JSON.stringify(shape)}`)
    assert.deepEqual(r.added, MANAGED_PLUGIN_NAMES, `shape: ${JSON.stringify(shape)}`)
    // All four land inside the plugins block, before model:.
    assert.ok(r.next.indexOf('- apexnodes-video-tools') < r.next.indexOf('model:'), `shape: ${JSON.stringify(shape)}`)
    assert.equal((r.next.match(/^plugins:/gm) || []).length, 1)
  }
})

test('ensurePluginsEnabledYaml restores enabled: under a plugins: block that lost it', () => {
  for (const shape of ['plugins:\n', 'plugins: {}\n']) {
    const r = ensurePluginsEnabledYaml(`${shape}model:\n  default: x\n`)
    assert.equal(r.changed, true, `shape: ${JSON.stringify(shape)}`)
    assert.deepEqual(r.added, MANAGED_PLUGIN_NAMES)
    assert.match(r.next, /plugins:\n {2}enabled:\n {4}- apex-overlay/)
    assert.ok(r.next.indexOf('- apexnodes-video-tools') < r.next.indexOf('model:'))
  }
})

test('ensurePluginsEnabledYaml handles PyYAML re-dump shapes (2-space items, flow list)', () => {
  // Block list re-dumped at the key's own indent.
  const redump = 'plugins:\n  enabled:\n  - apex-overlay\n  - user-extra\nmodel:\n  default: x\n'
  const r = ensurePluginsEnabledYaml(redump)
  assert.equal(r.changed, true)
  assert.deepEqual(r.added, ['apexnodes-douyin-tools', 'apexnodes-social-tools', 'apexnodes-video-tools'])
  // Missing names appended after the last item, at the SAME 2-space indent.
  assert.match(r.next, /- user-extra\n {2}- apexnodes-douyin-tools\n {2}- apexnodes-social-tools\n {2}- apexnodes-video-tools\nmodel:/)

  // Flow list: rewritten as a block list, existing entries + order kept.
  const flow = 'plugins:\n  enabled: [apex-overlay, user-extra]\nmodel:\n  default: x\n'
  const r2 = ensurePluginsEnabledYaml(flow)
  assert.equal(r2.changed, true)
  assert.match(r2.next, /enabled:\n {4}- apex-overlay\n {4}- user-extra\n {4}- apexnodes-douyin-tools/)
  assert.equal((r2.next.match(/- apex-overlay/g) || []).length, 1)
})

test('ensurePluginsEnabledYaml tolerates comments/quotes in the list (example-copy shape)', () => {
  const raw =
    'plugins:\n' +
    '  enabled:\n' +
    '    # ApexNodes cloud overlay boot hook. Applies our apex_overlay seams\n' +
    '    # onto upstream Hermes at load time.\n' +
    "    - 'apex-overlay'  # pinned\n" +
    '\n' +
    'model:\n' +
    '  default: x\n'
  const r = ensurePluginsEnabledYaml(raw)
  assert.equal(r.changed, true)
  // Quoted + commented entry recognized: apex-overlay NOT re-added.
  assert.deepEqual(r.added, ['apexnodes-douyin-tools', 'apexnodes-social-tools', 'apexnodes-video-tools'])
  // Comments survive; insertion right after the last existing item.
  assert.match(r.next, /# ApexNodes cloud overlay boot hook/)
  assert.match(r.next, /- 'apex-overlay' {2}# pinned\n {4}- apexnodes-douyin-tools/)
})

test('ensurePluginsEnabledYaml never touches structurally unexpected shapes', () => {
  // Inline non-empty flow map — too exotic for line surgery; leave alone.
  const exotic = 'plugins: {enabled: [apex-overlay]}\nmodel:\n  default: x\n'
  const r = ensurePluginsEnabledYaml(exotic)
  assert.equal(r.changed, false)
  assert.equal(r.next, exotic)
  // Empty managed list → no-op.
  assert.equal(ensurePluginsEnabledYaml('model: {}\n', []).changed, false)
})

// --- skills.disabled healer (hc-406 v0.18 upgrade path) ---
// A config seeded under v0.17 keeps its old skills.disabled after a runtime
// bump to v0.18; ensureSkillsDisabledYaml unions the newly-graded-OFF names in
// so freshly-bundled skills we cut (huggingface-hub / maps / plan) don't ship
// active. Add-only: user enable-toggles survive; nothing is ever removed.

test('ensureSkillsDisabledYaml unions the v0.18-new disabled names into a v0.17 config', () => {
  // A minimal v0.17-shaped block missing the three new names + carrying the two
  // now-dead orphans and a user enable (notion removed to turn it on).
  const v017 =
    'skills:\n' +
    '  disabled:\n' +
    '    - google-workspace\n' +
    '    - xurl\n' +
    '    - kanban-orchestrator\n' +
    '    - kanban-worker\n' +
    'model:\n' +
    '  default: x\n'
  const r = ensureSkillsDisabledYaml(v017)
  assert.equal(r.changed, true)
  // The three new bundled-OFF skills are added…
  for (const n of ['huggingface-hub', 'maps', 'plan']) {
    assert.ok(r.added.includes(n), `should add ${n}`)
    assert.ok(r.next.includes(`\n    - ${n}\n`), `should list ${n}`)
  }
  // …existing entries (incl. the dead orphans) are preserved, not removed…
  assert.match(r.next, /- kanban-orchestrator\n {4}- kanban-worker/)
  // …inserted INSIDE the skills block, before the next top-level key.
  assert.ok(r.next.indexOf('- plan') < r.next.indexOf('model:'))
  // …and no duplicates.
  for (const n of ['huggingface-hub', 'maps', 'plan', 'google-workspace']) {
    assert.equal((r.next.match(new RegExp(`- ${n}\\b`, 'g')) || []).length, 1, `duplicated: ${n}`)
  }
})

test('ensureSkillsDisabledYaml never re-adds a name the user removed to enable a skill', () => {
  // The full managed list minus one name (`powerpoint` was never disabled; the
  // user here removed `maps` to enable it). Add-only means the healer WILL
  // re-add `maps` — so to prove "respect a user enable" we pass a custom wanted
  // list that omits it, mirroring how a future reclass would drop a name.
  const seeded = seedSkillsBlockYaml(['google-workspace', 'xurl'])
  // User turned xurl back on (removed it):
  const userEdited = seeded.replace('    - xurl\n', '')
  // Reconcile against the SAME reduced managed list → nothing missing → no-op.
  const r = ensureSkillsDisabledYaml(userEdited, ['google-workspace'])
  assert.equal(r.changed, false)
  assert.equal(r.next, userEdited)
})

test('ensureSkillsDisabledYaml appends the whole block when skills: is absent', () => {
  const raw = 'model:\n  default: x\nplugins:\n  enabled:\n    - apex-overlay\n'
  const r = ensureSkillsDisabledYaml(raw)
  assert.equal(r.changed, true)
  assert.deepEqual(r.added, SEED_DISABLED_SKILLS)
  assert.ok(r.next.startsWith(raw)) // append-only
  assert.match(r.next, /^skills:\n {2}disabled:\n/m)
  assert.equal((r.next.match(/^skills:/gm) || []).length, 1)
})

test('ensureSkillsDisabledYaml is idempotent on a fresh full seed (no-op)', () => {
  const seed = seedSkillsBlockYaml()
  const r = ensureSkillsDisabledYaml(seed)
  assert.equal(r.changed, false)
  assert.equal(r.next, seed)
  assert.deepEqual(r.added, [])
})

test('ensureSkillsDisabledYaml handles PyYAML re-dump shapes (2-space items, [], null)', () => {
  const redump = 'skills:\n  disabled:\n  - google-workspace\n  - user-kept\nmodel:\n  default: x\n'
  const r = ensureSkillsDisabledYaml(redump, ['google-workspace', 'maps'])
  assert.equal(r.changed, true)
  assert.deepEqual(r.added, ['maps'])
  assert.match(r.next, /- user-kept\n {2}- maps\nmodel:/)

  for (const shape of ['  disabled: []\n', '  disabled: null\n', '  disabled:\n']) {
    const r2 = ensureSkillsDisabledYaml(`skills:\n${shape}model:\n  default: x\n`, ['maps', 'plan'])
    assert.equal(r2.changed, true, `shape: ${JSON.stringify(shape)}`)
    assert.deepEqual(r2.added, ['maps', 'plan'])
  }
  // Empty managed list → no-op.
  assert.equal(ensureSkillsDisabledYaml('skills:\n  disabled: []\n', []).changed, false)
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

// --- syncCustomProviderKeyYaml ---

const ROTATED_CONFIG = [
  'model:',
  '  api_key: sk-fresh',
  '  base_url: https://apex-nodes.com/relay/v1',
  '  default: deepseek-v4-pro-APEX',
  '  provider: custom',
  'custom_providers:',
  '- api_key: sk-stale',
  '  base_url: https://apex-nodes.com/relay/v1',
  '  model: deepseek-v4-pro-APEX',
  '  name: Apex-nodes.com',
  'skills:',
  '  disabled: []',
  ''
].join('\n')

test('syncCustomProviderKeyYaml refreshes a rotated relay entry key (PyYAML dump shape)', () => {
  const { changed, next } = syncCustomProviderKeyYaml(ROTATED_CONFIG, 'https://apex-nodes.com/relay/v1/', 'sk-fresh')
  assert.equal(changed, true)
  assert.match(next, /- api_key: sk-fresh\n {2}base_url: https:\/\/apex-nodes\.com\/relay\/v1/)
  // model.* block and other keys untouched.
  assert.match(next, /model:\n {2}api_key: sk-fresh/)
  assert.match(next, /name: Apex-nodes\.com/)
})

test('syncCustomProviderKeyYaml is a no-op when the key already matches or nothing matches', () => {
  const synced = syncCustomProviderKeyYaml(ROTATED_CONFIG, 'https://apex-nodes.com/relay/v1', 'sk-stale')
  assert.equal(synced.changed, false)
  assert.equal(synced.next, ROTATED_CONFIG)

  const otherBase = syncCustomProviderKeyYaml(ROTATED_CONFIG, 'https://elsewhere.example/v1', 'sk-fresh')
  assert.equal(otherBase.changed, false)

  const noList = syncCustomProviderKeyYaml('model:\n  api_key: sk-a\n', 'https://apex-nodes.com/relay/v1', 'sk-b')
  assert.equal(noList.changed, false)

  assert.equal(syncCustomProviderKeyYaml('', 'https://apex-nodes.com/relay/v1', 'sk-b').changed, false)
})

test('syncCustomProviderKeyYaml only touches the matching entry in a multi-entry list', () => {
  const multi = [
    'custom_providers:',
    '- api_key: sk-other',
    '  base_url: https://my-own-endpoint.example/v1',
    '  name: mine',
    '- api_key: sk-stale',
    '  base_url: https://apex-nodes.com/relay/v1',
    '  name: Apex-nodes.com',
    ''
  ].join('\n')
  const { changed, next } = syncCustomProviderKeyYaml(multi, 'https://apex-nodes.com/relay/v1', 'sk-fresh')
  assert.equal(changed, true)
  assert.match(next, /- api_key: sk-other\n {2}base_url: https:\/\/my-own-endpoint\.example\/v1/)
  assert.match(next, /- api_key: sk-fresh\n {2}base_url: https:\/\/apex-nodes\.com\/relay\/v1/)
})
