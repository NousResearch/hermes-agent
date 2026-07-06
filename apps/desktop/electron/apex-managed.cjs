/**
 * apex-managed.cjs
 *
 * Pure, electron-free helpers for the ApexNodes "managed LLM" default path
 * (Desktop V0.2). Kept standalone (no `require('electron')`) so it can be
 * unit-tested with `node --test`, same pattern as connection-config.cjs /
 * dashboard-token.cjs. main.cjs requires these and wires them into the
 * electron-coupled IPC + boot layer.
 *
 * Background — the two independent "login/connection" systems this app has:
 *
 *   1. REMOTE GATEWAY (connection-config.cjs / dashboard-token.cjs /
 *      oauth-net-request.cjs / gateway-ws-probe.cjs): connects the desktop to a
 *      *remote orchestration backend* (the dashboard/gateway that runs agent
 *      sessions) via HttpOnly session cookies + /api/ws tickets. This is a
 *      power-user feature, OFF by default; the desktop runs a LOCAL runtime.
 *
 *   2. MANAGED LLM (this module): points the LOCAL runtime's inference at the
 *      ApexNodes relay so a signed-in user gets zero-key chat. The relay is an
 *      OpenAI-compatible endpoint that bills the user's cloud account; the
 *      runtime just needs `model.base_url` + `model.api_key` + `model.default`
 *      in config.yaml (the exact same three fields the "Local / custom
 *      endpoint" BYOK flow already writes via /api/model/set).
 *
 * These are different concerns with different credentials — this module is ONLY
 * about (2). It never touches the remote-gateway cookie machinery.
 *
 * ── Backend contract (P0, confirmed) ────────────────────────────────────────
 * The relay validates an *agent-scoped* key from the cloud `api_keys` table; the
 * raw key is minted server-side and is never returned by the generic APIs (login
 * yields a JWT; the dashboard "API Keys" page mints a `user_api_keys` capability
 * key the relay does NOT accept). So a dedicated provisioning endpoint mints a
 * relay-VALID key for a logged-in user:
 *
 *   POST {API_BASE}/api/v1/desktop/provision-key
 *   Authorization: Bearer <login JWT>
 *   body: {}
 *   200 → { api_key, base_url, model }
 *
 * The desktop uses base_url + model FROM THE RESPONSE (never hardcoded) so the
 * server stays the source of truth for routing; the local DEFAULT_* constants
 * are only a fallback for display/seed when no response is on hand. If the
 * endpoint isn't reachable (not deployed, network error),
 * `resolveManagedRelayCredential` returns '' and the desktop transparently
 * falls back to the BYOK onboarding (no regression).
 */

// ── ApexNodes default endpoints ─────────────────────────────────────────────
// All overridable via env so a staging build can retarget without a code change
// (mirrors how main.cjs lets HERMES_DESKTOP_* env vars override prod defaults).

// User-facing site host. Login lives under `${AUTH_BASE}/api/v1/auth/...`. The
// relay public path also lives on this host (nginx `location /relay/` →
// 127.0.0.1:7000, prefix stripped).
const DEFAULT_AUTH_BASE = 'https://apex-nodes.com'

// API host for machine endpoints, incl. the desktop key-provisioning route
// (`POST {API_BASE}/api/v1/desktop/provision-key`). Both hosts route /api/ to
// the same backend, but the contract pins provision-key on the api. host.
const DEFAULT_API_BASE = 'https://api.apex-nodes.com'

// Fallback relay public base_url for `model.base_url` — used only for the
// display/seed when no provision-key response is on hand; the live path uses the
// base_url returned by provision-key. The relay's OpenAI-compatible chat route
// is `/v1/chat/completions`; nginx strips the `/relay` prefix, so the public
// form that reaches it is `https://apex-nodes.com/relay/v1/chat/completions`.
// The runtime appends `/chat/completions`, hence the base ends at `/relay/v1`.
const DEFAULT_RELAY_BASE_URL = 'https://apex-nodes.com/relay/v1'

// Real model the relay routes to (our master key). hc-184 decouples the routed
// model from the `model` field the runtime sends — the relay routes by DB truth
// (verified: the relay ignores the request's `model` entirely and returns
// `deepseek-v4-pro` for ANY value, including unknown ids), so the model id we
// write to config is cosmetic to the relay.
const DEFAULT_MANAGED_MODEL = 'deepseek-v4-pro'

// The model id we actually WRITE to config.yaml (`model.default` + the
// custom_providers entry's `model`) and show in the UI.
//
// ⚠️ This must be a name that is NOT an exact id in any built-in provider's
// static model catalog (`hermes_cli/models.py` `_PROVIDER_MODELS`). The bare
// routed id `deepseek-v4-pro` IS in the built-in DeepSeek catalog, and that is
// exactly what broke managed chat:
//
//   The desktop runs `hermes dashboard`; its embedded chat builds the agent via
//   `tui_gateway/server.py::_make_agent`. At boot (no per-session override) that
//   path resolves the model through `_resolve_startup_runtime`, which — when an
//   inference-model env hint is present (`HERMES_MODEL`/`HERMES_INFERENCE_MODEL`,
//   set by the runtime's own launcher and inheritable into the backend) — runs
//   `detect_static_provider_for_model(<model id>, …)`. That does an EXACT match
//   against the built-in catalogs (`models.py:1885`), so `deepseek-v4-pro`
//   resolves to provider `deepseek` (and `kimi-k2.6`→`kimi-coding`,
//   `glm-5.2`→`zai`), OVERRIDING the configured `provider: custom`. The built-in
//   provider has no key → `agent/agent_init.py` raises "Provider 'deepseek' is
//   set in config.yaml but no API key was found." The gateway caches that failed
//   build (`agent_build_started`), so switching models in the picker can't
//   recover the session — every selection shows the same sticky boot error. The
//   ONLY fix is a boot config whose model id does not collide.
//
// `deepseek-v4-pro-APEX` (the ApexNodes display name) is collision-free
// (`detect_static_provider_for_model` returns None — verified) AND relay-valid
// (HTTP 200, routed to deepseek-v4-pro — verified). Using it as the config model
// id makes the startup path resolve to the relay in every case (with or without
// the env hint), proven against the runtime venv.
const MANAGED_MODEL_DISPLAY = 'deepseek-v4-pro-APEX'

// The runtime treats the relay as a generic OpenAI-compatible endpoint, so the
// provider slug is the same `custom` the local/custom BYOK flow uses. Reusing
// `custom` means zero new runtime provider plumbing.
const MANAGED_PROVIDER = 'custom'

// Display name of the relay's `custom_providers:` entry. The runtime groups
// custom endpoints by this name in its model picker (users see an
// "APEX-NODES.COM" group), and Hermes' own writer
// (`hermes_cli/main.py::_save_custom_provider`) uses the exact same
// `{name, base_url, api_key, model}` entry shape. We register this entry so the
// relay is a *named* custom provider — the format Hermes produces after a
// `hermes model` custom-endpoint selection — which keeps the endpoint durable
// across `/model` picker switches and session resume (those persist
// `provider: custom:<slug>`, which only resolves when the named entry exists).
// The collision fix itself is the non-colliding model id above; this entry is
// the native-format hardening that goes with it.
const MANAGED_PROVIDER_NAME = 'Apex-nodes.com'

// ── ApexNodes China default profile (hc-392) ───────────────────────────────
// The desktop pre-seeds config.yaml BEFORE install.sh can copy
// cli-config.yaml.example (seedDefaultModelConfig in main.cjs only writes when
// config.yaml is absent, and install.sh's example-copy is likewise
// absent-gated — so the seed wins). The China profile therefore CANNOT rely on
// cli-config.yaml.example reaching the desktop; we fold the same two policy
// lists into the seed here so skill-cut + Copilot-disable actually take effect
// on a fresh desktop install. The runtime reads these from config.yaml via
// agent.skill_utils.get_disabled_skill_names() and
// hermes_cli.model_switch.list_authenticated_providers (model.disabled_providers).
//
// Providers never probed / live-fetched / shown. Matched case-insensitively
// against the Hermes slug + its models.dev id.
const MODEL_DISABLED_PROVIDERS = ['copilot']

// Standalone runtime plugins the product REQUIRES enabled. The runtime's
// plugin loader is opt-in: only names listed under `plugins.enabled` in
// config.yaml load (bundled backend/platform plugins auto-load regardless), so
// a config.yaml WITHOUT this block ships every standalone plugin disabled —
// including apex-overlay, whose seams enforce the hc-392 provider denylist.
// MUST stay in sync with the `plugins.enabled` list in cli-config.yaml.example
// (the pure-CLI path; the three apexnodes-* tool entries land there with
// feat/plugins-gateway-p1).
const MANAGED_PLUGIN_NAMES = [
  // ApexNodes cloud overlay boot hook — applies the apex_overlay seams onto
  // upstream Hermes at load time (monkey-patch, zero in-place edits).
  'apex-overlay',
  // ApexNodes platform tools (P1): social-media data / download+transcribe /
  // text-to-video, routed through the platform tools gateway.
  'apexnodes-douyin-tools',
  'apexnodes-social-tools',
  'apexnodes-video-tools'
]

// Skills physically present in ~/.hermes/skills/ but kept INACTIVE by default
// (never loaded until removed from this list in Settings → Skills). We disable
// rather than delete so upstream merges stay painless. Names below are the
// SKILL.md frontmatter `name:` (which the toggle matches, case-sensitive) —
// note the four that differ from their folder names (serving-llms-vllm,
// evaluating-llms-harness, segment-anything-model, audiocraft-audio-generation).
// MUST stay in sync with the `skills.disabled` list in cli-config.yaml.example
// (the pure-CLI path).
//
// ── hc-406 v0.18 全集重分级 ──────────────────────────────────────────────────
// The runtime's bundled `skills/` (72 skills @ v0.18) auto-copy to
// ~/.hermes/skills/ and are ACTIVE unless listed here; the ~101 opt-in
// `optional-skills/` are NOT copied at setup (installed on demand via
// `hermes skills install`), so they are already-off and never need seeding.
// Grading matrix + full 72-skill table: hermes-cloud
// docs/DESKTOP-CHINA-SKILL-MATRIX.md. Judge = "does the skill depend ONLY on
// (a) local exec or (b) a CN-reachable, stable service?" — yes → active; no →
// disabled here (a 30s-timeout skill reads as "the agent is broken").
//
// v0.18 delta vs v0.17: +3 disabled (huggingface-hub / maps / plan — all
// new-bundled and C/dev-niche), −2 removed dead orphans (kanban-orchestrator /
// kanban-worker — no longer a bundled skill; the successor is the opt-in
// optional-skills/creative/kanban-video-orchestrator). The 22 newly-bundled
// A/B skills we deliberately KEEP ACTIVE (computer-use, apple/*, powerpoint,
// obsidian, ocr-and-documents, baoyu-infographic, yuanbao, …) are simply
// absent from this list.
const SEED_DISABLED_SKILLS = [
  // ── D — walled / geo-blocked / competitor (无 VPN 必超时挂起) ──────────────
  'google-workspace', 'xurl', 'youtube-content', 'polymarket',
  'teams-meeting-pipeline', 'claude-code', 'codex', 'gif-search',
  // ── C — CN-slow / needs mirror or 国产源 before it's stable ────────────────
  // Cross-border SaaS (reachable-but-slow / needs foreign account):
  'notion', 'airtable',
  // github.com reachable-but-slow, raw.githubusercontent.com blocked:
  'github-auth', 'github-code-review', 'github-issues',
  'github-pr-workflow', 'github-repo-management',
  // HuggingFace / heavy-MLOps model discovery (HF blocked/slow; ModelScope
  // route is the V0.2 localization roadmap — HF_ENDPOINT mirror is seeded but
  // these stay off as non-essential for a CN desktop assistant):
  'huggingface-hub', 'serving-llms-vllm', 'llama-cpp', 'weights-and-biases',
  'evaluating-llms-harness', 'segment-anything-model', 'comfyui',
  'audiocraft-audio-generation', 'heartmula',
  // OSM/OSRM: CN-slow + GCJ-02 ~500m offset + weak CN routing (AMap needed):
  'maps',
  // arXiv: export.arxiv.org reachable-but-slow/unstable:
  'arxiv',
  // ── DEV-B — capable but dev-niche; product-focus off (toggle on per-need) ──
  // Coding-agent / self-config:
  'opencode', 'hermes-agent', 'hermes-agent-skill-authoring',
  // Dev workflow / debug / review:
  'codebase-inspection', 'simplify-code', 'test-driven-development',
  'systematic-debugging', 'requesting-code-review', 'node-inspect-debugger',
  'python-debugpy', 'jupyter-live-kernel', 'plan', 'spike',
  // Niche integrations / creative / research (local but seldom-needed):
  'himalaya', 'design-md', 'research-paper-writing', 'pretext',
  'songwriting-and-ai-music', 'songsee', 'manim-video', 'p5js',
  'touchdesigner-mcp', 'openhue',
  // Internal / QA:
  'dogfood'
]

// ── hc-406: platform search / extraction gateway seed (待 S2/S3 上线) ─────────
// The China desktop epic replaces the walled ddgs web-search (hc-408) and adds a
// self-hosted Firecrawl web_extract leg (hc-414). Both are桌面-side config-only
//接入 (zero overlay) — the runtime reads a top-level `web:` block + two env
// vars. S2/S3's WORK-NOTES now pin the EXACT shape (hermes-cloud
// docs/work-notes/WORK-NOTES-hc408.md ⭐ + WORK-NOTES-hc414.md §4):
//
//   web:
//     search_backend: searxng      # hc-408 — only overrides search
//     extract_backend: firecrawl   # hc-414 — only overrides extract
//   env: SEARXNG_URL=https://api.apex-nodes.com/api/v1/search/searxng
//        FIRECRAWL_API_URL=<公网 Firecrawl 走向 — 待定>
//   (SEARXNG_URL has NO trailing /search — the fork provider appends it.)
//
// NOT emitted into the seed yet — two prerequisites are unmet, and seeding a
// backend whose endpoint is absent would break desktop search/extract on every
// fresh install:
//   1. hc-408 (feat/hc408-relay-search) is NOT merged to cloud main → the
//      `/api/v1/search/searxng/search` route is not on prod yet.
//   2. hc-414 explicitly leaves the *public* Firecrawl URL undecided (§4: cloud
//      Firecrawl binds 127.0.0.1 only; a nginx-fronted public走向 is a Kael/PM
//      decision) → there is no FIRECRAWL_API_URL to write, and web_extract for
//      desktop is opt-in ("若桌面暂不开 web_extract,S6 可跳过").
//
// TODO(hc408/hc414 seed activation): once both endpoints are live on prod, lift
// the block above into a SEED_WEB_GATEWAY constant + a seedWebGatewayBlockYaml()
// helper (top-level `web:` key, same pattern as seedSkillsBlockYaml), fold it
// into seedDefaultModelConfig's composition, add the two env vars to the desktop
// shell env (SEARXNG_URL alongside HF_ENDPOINT in backend-env.cjs), and add a
// guard/heal pass in main.cjs guardConfigYamlProductBlocks (union the `web:`
// keys, mirroring ensureSkillsDisabledYaml) so the upgrade path covers it too.
// The env vars ride the same spawn merge (`{...process.env, ...backend.env}`).

/**
 * Render the `model.disabled_providers` YAML lines (indented to sit INSIDE the
 * `model:` block). Returns '' when the list is empty. Kept as a helper so the
 * managed (apex-managed) and BYOK (main.cjs raw string) seed paths emit the
 * identical block.
 *
 * @param {string[]} [providers]
 * @returns {string}
 */
function modelDisabledProvidersYaml(providers = MODEL_DISABLED_PROVIDERS) {
  const list = Array.isArray(providers) ? providers.filter(p => String(p || '').trim()) : []
  if (!list.length) return ''
  let yaml = '  disabled_providers:\n'
  for (const p of list) yaml += `    - ${String(p).trim()}\n`
  return yaml
}

/**
 * Render the top-level `skills.disabled` YAML block. Returns '' when empty.
 * Top-level `skills:` key — no collision with the `model:` / `display:` blocks
 * the seed already emits.
 *
 * @param {string[]} [skills]
 * @returns {string}
 */
function seedSkillsBlockYaml(skills = SEED_DISABLED_SKILLS) {
  const list = Array.isArray(skills) ? skills.filter(s => String(s || '').trim()) : []
  if (!list.length) return ''
  let yaml =
    '# ApexNodes China default profile (hc-392): skills shipped but OFF by\n' +
    '# default. Toggle any on in Settings → Skills. Files are kept (not\n' +
    '# deleted) so upstream merges stay clean.\n' +
    'skills:\n' +
    '  disabled:\n'
  for (const s of list) yaml += `    - ${String(s).trim()}\n`
  return yaml
}

/**
 * Render the top-level `plugins.enabled` YAML block for the desktop seed.
 * Mirrors the block cli-config.yaml.example carries: the runtime's standalone
 * plugin loader is opt-in, so a seeded config.yaml without this block would
 * disable apex-overlay + the apexnodes-* tool plugins on every fresh desktop
 * install (the seed pre-empts install.sh's example-copy — both absent-gated,
 * seed runs first). Returns '' when the list is empty.
 *
 * @param {string[]} [plugins]
 * @returns {string}
 */
function seedPluginsBlockYaml(plugins = MANAGED_PLUGIN_NAMES) {
  const list = Array.isArray(plugins) ? plugins.filter(p => String(p || '').trim()) : []
  if (!list.length) return ''
  let yaml =
    '# Standalone runtime plugins are opt-in: only names listed here load.\n' +
    '# apex-overlay carries the ApexNodes seams (incl. the provider denylist\n' +
    '# above); the apexnodes-* tools ride the platform tools gateway.\n' +
    'plugins:\n' +
    '  enabled:\n'
  for (const p of list) yaml += `    - ${String(p).trim()}\n`
  return yaml
}

/**
 * Ensure every managed plugin name is present under `plugins.enabled` in a
 * raw config.yaml. Pure line surgery (no YAML round-trip — comments and
 * formatting survive), tolerant of the shapes we actually meet: the desktop /
 * example seed (4-space list indent, comments inside the list), PyYAML
 * re-dumps (2-space list items, `enabled: []` / flow lists / `null`), and
 * hand edits.
 *
 * ADD-ONLY by contract: user-added extra plugins and their order are always
 * preserved; when every managed name is already present the input is returned
 * unchanged (idempotent), and anything structurally unexpected → no change
 * (same philosophy as syncCustomProviderKeyYaml — never corrupt the file).
 *
 * @param {string} raw config.yaml contents
 * @param {string[]} [plugins]
 * @returns {{ changed: boolean, next: string, added: string[] }}
 */
function ensurePluginsEnabledYaml(raw, plugins = MANAGED_PLUGIN_NAMES) {
  return ensureListBlockYaml(raw, {
    blockKey: 'plugins',
    listKey: 'enabled',
    wanted: plugins,
    seedBlock: seedPluginsBlockYaml
  })
}

/**
 * Ensure every managed skill name is present under `skills.disabled` in a raw
 * config.yaml — the UPGRADE path for the hc-406 v0.18 reclassification. A
 * fresh seed already carries the full list (seedDefaultModelConfig), but an
 * install seeded under v0.17 keeps its old 49-name `skills.disabled` after a
 * runtime bump to v0.18; without this, the newly-bundled skills we graded OFF
 * (huggingface-hub / maps / plan) would ship ACTIVE on that upgraded machine.
 *
 * Same ADD-ONLY contract as the plugins healer: user toggles (a name the user
 * REMOVED to enable a skill) are never re-added out from under them — this only
 * unions in managed names that are wholly absent, and never removes anything
 * (so stale entries like the dropped kanban-* orphans are left in place,
 * harmless: they match no skill). Idempotent; structurally unexpected → no change.
 *
 * @param {string} raw config.yaml contents
 * @param {string[]} [skills]
 * @returns {{ changed: boolean, next: string, added: string[] }}
 */
function ensureSkillsDisabledYaml(raw, skills = SEED_DISABLED_SKILLS) {
  return ensureListBlockYaml(raw, {
    blockKey: 'skills',
    listKey: 'disabled',
    wanted: skills,
    seedBlock: seedSkillsBlockYaml
  })
}

/**
 * Generic add-only union of `wanted` names into a top-level `${blockKey}:` →
 * `${listKey}:` YAML list, by pure line surgery (no YAML round-trip — comments
 * and formatting survive). Backs both ensurePluginsEnabledYaml and
 * ensureSkillsDisabledYaml (identical block/list/list-of-scalars shape).
 * Tolerant of the shapes we actually meet: the desktop / example seed (4-space
 * list indent, comments inside the list), PyYAML re-dumps (2-space list items,
 * `${listKey}: []` / flow lists / `null`), and hand edits.
 *
 * ADD-ONLY by contract: user-added entries and their order are always
 * preserved; when every wanted name is already present the input is returned
 * unchanged (idempotent), and anything structurally unexpected → no change
 * (same philosophy as syncCustomProviderKeyYaml — never corrupt the file).
 *
 * @param {string} raw
 * @param {{ blockKey: string, listKey: string, wanted: string[], seedBlock: (names: string[]) => string }} opts
 * @returns {{ changed: boolean, next: string, added: string[] }}
 */
function ensureListBlockYaml(raw, { blockKey, listKey, wanted: wantedRaw, seedBlock }) {
  const source = String(raw || '')
  const wanted = (Array.isArray(wantedRaw) ? wantedRaw : []).map(p => String(p || '').trim()).filter(Boolean)
  const unchanged = { changed: false, next: source, added: [] }
  if (!wanted.length) return unchanged

  const unquote = value => String(value || '').trim().replace(/^(["'])(.*)\1$/, '$2')
  // `- name  # comment` → name (a trailing comment needs whitespace before #).
  const itemName = text => unquote(String(text).replace(/\s+#.*$/, ''))
  const blockRe = new RegExp(`^${blockKey}:`)

  const lines = source.split('\n')
  let blockLine = -1
  for (let i = 0; i < lines.length; i++) {
    if (blockRe.test(lines[i])) { blockLine = i; break }
  }

  // ── no top-level block: key at all → append the full seed block ─────────
  if (blockLine < 0) {
    const next = source.replace(/\n*$/, '\n') + seedBlock(wanted)
    return { changed: true, next, added: wanted.slice() }
  }

  const blockRest = lines[blockLine].slice(`${blockKey}:`.length).trim()
  if (blockRest && !blockRest.startsWith('#')) {
    // Inline value. `${blockKey}: {}` (PyYAML's empty-map dump) is safely
    // replaceable with the block form; any other inline shape is unexpected.
    if (!/^\{\s*\}(\s*#.*)?$/.test(blockRest)) return unchanged
    const replacement = [`${blockKey}:`, `  ${listKey}:`]
    for (const name of wanted) replacement.push(`    - ${name}`)
    lines.splice(blockLine, 1, ...replacement)
    return { changed: true, next: lines.join('\n'), added: wanted.slice() }
  }

  // ── find the end of the block and its `${listKey}:` key ─────────────────
  let blockEnd = lines.length
  let listLine = -1
  let listIndent = ''
  let childIndent = ''
  for (let i = blockLine + 1; i < lines.length; i++) {
    const line = lines[i]
    if (/^\S/.test(line)) { blockEnd = i; break } // next top-level key
    const key = line.match(/^(\s+)([A-Za-z0-9_-]+):(.*)$/)
    if (key && !childIndent) childIndent = key[1]
    if (key && key[2] === listKey && listLine < 0) {
      listLine = i
      listIndent = key[1]
    }
  }

  if (listLine < 0) {
    // block exists but the list key is gone — insert one at the top of the
    // block, matching the block's child indent when it has one.
    const indent = childIndent || '  '
    const insert = [`${indent}${listKey}:`]
    for (const name of wanted) insert.push(`${indent}  - ${name}`)
    lines.splice(blockLine + 1, 0, ...insert)
    return { changed: true, next: lines.join('\n'), added: wanted.slice() }
  }

  const listMatch = lines[listLine].match(new RegExp(`^\\s+${listKey}:(.*)$`))
  const listRest = (listMatch ? listMatch[1] : '').trim()
  if (listRest && !listRest.startsWith('#')) {
    // Inline value: [] / null / a flow list. Rewrite as a block list keeping
    // existing entries + order; anything else unexpected → no change.
    let existing
    if (/^\[\s*\]$/.test(listRest) || /^(~|null)$/i.test(listRest)) {
      existing = []
    } else if (/^\[.*\]$/.test(listRest)) {
      existing = listRest.slice(1, -1).split(',').map(itemName).filter(Boolean)
    } else {
      return unchanged
    }
    const missing = wanted.filter(name => !existing.includes(name))
    if (!missing.length) return unchanged
    const replacement = [`${listIndent}${listKey}:`]
    for (const name of existing.concat(missing)) replacement.push(`${listIndent}  - ${name}`)
    lines.splice(listLine, 1, ...replacement)
    return { changed: true, next: lines.join('\n'), added: missing }
  }

  // ── block list: collect existing item names + the last item line ────────
  const existing = []
  let lastItemLine = -1
  let itemIndent = ''
  for (let i = listLine + 1; i < blockEnd; i++) {
    const line = lines[i]
    if (!line.trim() || /^\s*#/.test(line)) continue // blanks/comments inside the list
    const item = line.match(/^(\s*)-\s+(.*)$/)
    if (item && item[1].length >= listIndent.length) {
      existing.push(itemName(item[2]))
      lastItemLine = i
      if (!itemIndent) itemIndent = item[1]
      continue
    }
    break // a sibling key (or anything else) ends the list
  }

  const missing = wanted.filter(name => !existing.includes(name))
  if (!missing.length) return unchanged
  const indent = itemIndent || `${listIndent}  `
  const insertAt = lastItemLine >= 0 ? lastItemLine + 1 : listLine + 1
  lines.splice(insertAt, 0, ...missing.map(name => `${indent}- ${name}`))
  return { changed: true, next: lines.join('\n'), added: missing }
}

// Endpoint paths. LOGIN_PATH / REGISTER_PATH are on AUTH_BASE; PROVISION_KEY_PATH
// is on API_BASE. GOOGLE_START_PATH is the backend's browser OAuth entry (on
// API_BASE — see the shared login-rework contract).
const LOGIN_PATH = '/api/v1/auth/login'
const REGISTER_PATH = '/api/v1/auth/register'
const PROVISION_KEY_PATH = '/api/v1/desktop/provision-key'
const GOOGLE_START_PATH = '/api/v1/auth/google/start'

// User-facing site path of the web login page. The desktop "用 APEX 登录" flow
// opens `${AUTH_BASE}/zh/login?desktop_cb=<loopback>&state=<s>`; the web login
// page honors desktop_cb and redirects the browser back to the loopback with the
// minted token. Locale-pinned to zh (the desktop is China-first).
const WEB_LOGIN_PATH = '/zh/login'

function trimTrailingSlash(value) {
  return String(value || '').replace(/\/+$/, '')
}

/**
 * Resolve the effective ApexNodes endpoints, applying env overrides. Pure: the
 * caller passes `process.env` (or a stub in tests) so this stays electron-free
 * and deterministic.
 *
 * @param {Record<string, string | undefined>} [env]
 * @returns {{ authBase: string, apiBase: string, relayBaseUrl: string,
 *             model: string, modelDisplay: string, provider: string,
 *             loginUrl: string, provisionKeyUrl: string }}
 */
function resolveApexEndpoints(env = {}) {
  const authBase = trimTrailingSlash(env.APEXNODES_AUTH_BASE || DEFAULT_AUTH_BASE)
  const apiBase = trimTrailingSlash(env.APEXNODES_API_BASE || DEFAULT_API_BASE)
  const relayBaseUrl = trimTrailingSlash(env.APEXNODES_RELAY_BASE_URL || DEFAULT_RELAY_BASE_URL)
  const model = String(env.APEXNODES_MANAGED_MODEL || DEFAULT_MANAGED_MODEL).trim() || DEFAULT_MANAGED_MODEL
  // The display id is what gets WRITTEN to config (collision-free with built-in
  // catalogs — see MANAGED_MODEL_DISPLAY). Precedence:
  //   1. explicit APEXNODES_MANAGED_MODEL_DISPLAY override
  //   2. when only APEXNODES_MANAGED_MODEL is overridden (e.g. staging), derive a
  //      collision-free id by appending the `-APEX` brand suffix to it
  //   3. the prod default display name
  const explicitDisplay = String(env.APEXNODES_MANAGED_MODEL_DISPLAY || '').trim()
  const modelDisplay =
    explicitDisplay ||
    (env.APEXNODES_MANAGED_MODEL ? `${model}-APEX` : MANAGED_MODEL_DISPLAY)

  return {
    authBase,
    apiBase,
    relayBaseUrl,
    model,
    modelDisplay,
    provider: MANAGED_PROVIDER,
    loginUrl: `${authBase}${LOGIN_PATH}`,
    registerUrl: `${authBase}${REGISTER_PATH}`,
    provisionKeyUrl: `${apiBase}${PROVISION_KEY_PATH}`
  }
}

/**
 * Build the browser start URL for "用 Google 登录" (Deliverable 2). The desktop
 * opens this in the system browser; the backend bounces through Google and
 * redirects to the loopback `redirect_uri` with `?token=<JWT>&state=<state>`.
 * Lives on API_BASE per the shared contract.
 *
 * @param {string} redirectUri  the loopback callback (http://127.0.0.1:<port>/cb)
 * @param {string} state        random CSRF token echoed back on the callback
 * @param {Record<string, string | undefined>} [env]
 * @returns {string}
 */
function googleStartUrl(redirectUri, state, env = {}) {
  const { apiBase } = resolveApexEndpoints(env)
  const u = new URL(`${apiBase}${GOOGLE_START_PATH}`)
  u.searchParams.set('redirect_uri', String(redirectUri || ''))
  u.searchParams.set('state', String(state || ''))
  return u.toString()
}

/**
 * Build the browser start URL for "用 APEX 登录" (Deliverable 3). Opens the web
 * login page with `desktop_cb` + `state`; the web page redirects the browser
 * back to the loopback with `?token=<access_token>&state=<state>` after a
 * successful login/register. Lives on AUTH_BASE (the user-facing site).
 *
 * @param {string} redirectUri  the loopback callback (http://127.0.0.1:<port>/cb)
 * @param {string} state        random CSRF token echoed back on the callback
 * @param {Record<string, string | undefined>} [env]
 * @returns {string}
 */
function apexWebLoginUrl(redirectUri, state, env = {}) {
  const { authBase } = resolveApexEndpoints(env)
  const u = new URL(`${authBase}${WEB_LOGIN_PATH}`)
  u.searchParams.set('desktop_cb', String(redirectUri || ''))
  u.searchParams.set('state', String(state || ''))
  return u.toString()
}

/**
 * Parse + validate a browser loopback callback request URL for the Google / APEX
 * flows. The browser is redirected to `http://127.0.0.1:<port>/cb?token=<JWT>&state=<s>`.
 * We require the path to be `/cb`, the `state` to match the one we generated
 * (CSRF defense), and a non-empty `token`. Anything else → { ok:false, ... } so
 * the loopback handler can respond with an error page and never apply a token.
 *
 * Pure: the caller passes the request URL (Node sets req.url to a path+query for
 * an http server, so we parse against a dummy origin) and the expected state.
 *
 * @param {string} requestUrl   req.url from the loopback server (path + query)
 * @param {string} expectedState the state we generated when starting the flow
 * @returns {{ ok: true, token: string } | { ok: false, reason: string, isCallback: boolean }}
 */
function parseLoopbackCallback(requestUrl, expectedState) {
  let parsed
  try {
    // req.url is path-relative; a dummy origin lets URL parse the query.
    parsed = new URL(String(requestUrl || ''), 'http://127.0.0.1')
  } catch {
    return { ok: false, reason: 'invalid_request', isCallback: false }
  }
  // Only the /cb path is the OAuth callback. Other paths (e.g. /favicon.ico the
  // browser auto-requests) must be ignored, not treated as a failed login.
  const isCallback = parsed.pathname === '/cb'
  if (!isCallback) {
    return { ok: false, reason: 'not_callback', isCallback: false }
  }
  const error = parsed.searchParams.get('error')
  if (error) {
    return { ok: false, reason: error, isCallback: true }
  }
  const state = parsed.searchParams.get('state') || ''
  const expected = String(expectedState || '')
  // Constant-ish comparison; states are random opaque tokens, lengths usually
  // equal, so a plain !== is acceptable here (no secret-dependent branch leak of
  // value, only of equality — same as the rest of the OAuth state checks).
  if (!expected || state !== expected) {
    return { ok: false, reason: 'state_mismatch', isCallback: true }
  }
  const token = (parsed.searchParams.get('token') || '').trim()
  if (!token) {
    return { ok: false, reason: 'missing_token', isCallback: true }
  }
  return { ok: true, token }
}

/**
 * True when a redirect/callback URL targets the loopback interface (127.0.0.1 /
 * ::1 / localhost) over http. The desktop only ever points the browser at its
 * own loopback; this guards against accidentally opening a non-loopback start
 * URL (defense in depth — the backend MUST also validate redirect_uri).
 *
 * @param {string} url
 * @returns {boolean}
 */
function isLoopbackUrl(url) {
  let parsed
  try {
    parsed = new URL(String(url || ''))
  } catch {
    return false
  }
  if (parsed.protocol !== 'http:') return false
  // URL normalizes an IPv6 host to its bracketed form ("[::1]"); strip the
  // brackets so the bare-address comparison matches.
  const host = parsed.hostname.toLowerCase().replace(/^\[|\]$/g, '')
  return host === '127.0.0.1' || host === '::1' || host === 'localhost'
}

/**
 * True when the managed-LLM default path is enabled for this install. Single
 * gate the rest of the app (boot seed, onboarding, IPC) consults so the flag
 * check isn't scattered.
 *
 * Shipped ON by default now that the backend provision-key endpoint is live on
 * prod (2026-06-27): managed is the zero-key default. Every path still
 * auto-degrades to BYOK if provision-key is ever unreachable, so this never
 * breaks chat. Set `APEXNODES_MANAGED=0` (or false/no/off) to force the legacy
 * BYOK-first build.
 *
 * @param {Record<string, string | undefined>} [env]
 */
function isManagedEnabled(env = {}) {
  const raw = String(env.APEXNODES_MANAGED ?? '').trim().toLowerCase()
  return raw !== '0' && raw !== 'false' && raw !== 'no' && raw !== 'off'
}

/**
 * Build the config.yaml `model:` block for the managed relay path. The runtime
 * resolver reads `model.base_url` / `model.api_key` directly from config and
 * honors them for a `custom` provider (see hermes_cli/web_server.py
 * `_apply_main_model_assignment` + the local-endpoint BYOK flow), so this is the
 * exact shape that routes inference through the relay.
 *
 * `base_url` and `model` come from the provision-key response when available
 * (server is the source of truth); the env/DEFAULT_* fallback is only for the
 * boot seed before we hold a response. Throws on a missing key — a managed block
 * without a credential would 401 every request, which is worse than falling back
 * to BYOK; callers gate on `resolveManagedRelayCredential` first.
 *
 * The written model id is the ApexNodes display name (`MANAGED_MODEL_DISPLAY`,
 * env-overridable), NOT the raw routed id — the raw `deepseek-v4-pro` collides
 * with the built-in DeepSeek catalog and gets the agent init mis-routed to the
 * keyless built-in `deepseek` provider (see MANAGED_MODEL_DISPLAY above). The
 * relay routes by DB truth and ignores this id, so it is safe + cosmetic on the
 * wire; locally it is the collision-free anchor that keeps resolution on the
 * relay's custom endpoint.
 *
 * Also returns a `custom_providers` entry registering the relay as a named
 * custom provider (`{name, base_url, api_key, model}` — Hermes' native shape).
 * The `model:` block keeps `provider: custom` + the relay `base_url`/`api_key`
 * (so the resolved provider class matches and there is no per-turn re-switch),
 * while the registered entry keeps the endpoint durable across picker switches /
 * session resume (which persist `provider: custom:<slug>`).
 *
 * @param {string} relayKey  the user's relay-valid cloud key
 * @param {Record<string, string | undefined>} [env]
 * @param {{ baseUrl?: string, model?: string }} [overrides] from provision-key
 * @returns {{
 *   default: string, provider: string, base_url: string, api_key: string,
 *   custom_providers: Array<{ name: string, base_url: string, api_key: string, model: string }>
 * }}
 */
function buildManagedModelConfig(relayKey, env = {}, overrides = {}) {
  const key = String(relayKey || '').trim()
  if (!key) {
    throw new Error('buildManagedModelConfig: a relay key is required.')
  }
  const endpoints = resolveApexEndpoints(env)
  // The model id WRITTEN to config must be collision-free with the built-in
  // catalogs (see MANAGED_MODEL_DISPLAY). The relay ignores the model id (routes
  // by DB truth), so a provision-key `overrides.model` is only honored when it is
  // ALREADY a non-colliding ApexNodes display id (ends with the `-APEX` brand
  // suffix); otherwise we use the display name so a raw routed id like
  // `deepseek-v4-pro` can never re-seed the collision the next time config is
  // (re)written at boot.
  const overrideModel = String(overrides.model || '').trim()
  const model = /-APEX$/i.test(overrideModel) ? overrideModel : endpoints.modelDisplay
  const baseUrl = trimTrailingSlash(overrides.baseUrl || '') || endpoints.relayBaseUrl
  return {
    default: model,
    provider: MANAGED_PROVIDER,
    base_url: baseUrl,
    api_key: key,
    // Register the relay as a named custom provider (Hermes-native shape) so the
    // endpoint stays durable across picker switches / resume. Same id as
    // model.default so both anchors agree.
    custom_providers: [
      {
        name: MANAGED_PROVIDER_NAME,
        base_url: baseUrl,
        api_key: key,
        model
      }
    ]
  }
}

/**
 * Parse + validate the provision-key response into the fields the desktop
 * persists and applies: { apiKey, baseUrl, model, email, name, plan }. Returns
 * null when the key is missing (caller falls back to BYOK). base_url/model fall
 * back to env defaults if the server omits them. email/name/plan are the
 * display-only identity the JWT-authed endpoint knows for certain — the
 * authoritative source for the account panel (a browser/Google sign-in JWT can
 * omit email, so the token claims alone aren't enough).
 *
 * @param {unknown} body  response of POST /api/v1/desktop/provision-key
 * @param {Record<string, string | undefined>} [env]
 * @returns {{ apiKey: string, baseUrl: string, model: string, email: string, name: string, plan: string } | null}
 */
function parseProvisionResponse(body, env = {}) {
  const key = relayKeyFromResponse(body)
  if (!key) return null
  const endpoints = resolveApexEndpoints(env)
  const obj = body && typeof body === 'object' ? body : {}
  const str = value => (typeof value === 'string' ? value.trim() : '')
  const baseUrl = typeof obj.base_url === 'string' ? trimTrailingSlash(obj.base_url) : ''
  const model = typeof obj.model === 'string' ? obj.model.trim() : ''
  return {
    apiKey: key,
    baseUrl: baseUrl || endpoints.relayBaseUrl,
    model: model || endpoints.model,
    email: str(obj.email),
    name: str(obj.name),
    plan: str(obj.plan)
  }
}

/**
 * Serialize the managed `model:` block (and the `custom_providers:` entry that
 * registers the relay) to a YAML snippet for seedDefaultModelConfig. Hand-rolled
 * (no yaml dep — this module is dependency-free like its siblings); values are
 * simple scalars (URL, slug, opaque key), but we double-quote the URL/key/name
 * defensively since they are opaque/external input.
 *
 * @param {{
 *   default: string, provider: string, base_url: string, api_key: string,
 *   custom_providers?: Array<{ name: string, base_url: string, api_key: string, model: string }>
 * }} block
 * @param {{ disabledProviders?: string[] }} [opts]  hc-392: when
 *   `disabledProviders` is given, its `disabled_providers:` lines are emitted
 *   INSIDE this `model:` block (a second top-level `model:` block would be a
 *   duplicate YAML key).
 * @returns {string}
 */
function managedModelConfigYaml(block, opts = {}) {
  const q = v => JSON.stringify(String(v)) // JSON string == valid YAML double-quoted scalar
  let yaml =
    'model:\n' +
    `  default: ${block.default}\n` +
    `  provider: ${block.provider}\n` +
    `  base_url: ${q(block.base_url)}\n` +
    `  api_key: ${q(block.api_key)}\n`
  if (opts && opts.disabledProviders) {
    yaml += modelDisabledProvidersYaml(opts.disabledProviders)
  }
  const entries = Array.isArray(block.custom_providers) ? block.custom_providers : []
  if (entries.length) {
    yaml += 'custom_providers:\n'
    for (const entry of entries) {
      yaml +=
        `  - name: ${q(entry.name)}\n` +
        `    base_url: ${q(entry.base_url)}\n` +
        `    api_key: ${q(entry.api_key)}\n` +
        `    model: ${entry.model}\n`
    }
  }
  return yaml
}

/**
 * Classify a credential-resolution result. The desktop should only switch the
 * default to managed when we actually hold a relay key; otherwise it must fall
 * back to BYOK. Centralizing the rule avoids each call site re-deriving it.
 *
 * @param {{ enabled: boolean, key: string | null }} state
 * @returns {'managed' | 'byok'}
 */
function defaultModelPath(state) {
  return state && state.enabled && typeof state.key === 'string' && state.key.trim() ? 'managed' : 'byok'
}

// ── Relay-key self-heal (401 → auto re-provision) ───────────────────────────
// provision-key ROTATES the relay key on every sign-in; the server marks the
// prior key `rotated` and only the newest is relay-valid. If the cloud rotates
// the active key out from under a signed-in desktop (e.g. a re-provision from
// another surface, or a background rotation), the on-disk key silently goes
// dead: the model picker's live `GET /v1/models` listing 401s and collapses to
// the single configured model — the "过几天列表缩水到只剩一个" bug. Manual
// re-login fixes it (provision re-mints + syncs), but the user shouldn't have
// to. These pure helpers back the auto-heal: main.cjs probes /v1/models at
// boot, and on a 401 re-runs the existing provision chain with the STORED login
// JWT (persisted encrypted alongside the relay key), then re-syncs the
// custom_providers entry. If the stored JWT is itself expired, provision-key
// 401s too → we stop and log (re-login UX is the existing sign-in flow's job).

/**
 * A relay HTTP status that means "this key is not accepted" — the trigger for a
 * re-provision. 401 (Invalid Agent API key) is the observed failure; 403 is
 * folded in defensively (a revoked/forbidden key is equally un-healable without
 * a fresh mint). Every other status (2xx, 5xx, network error) is NOT an auth
 * failure and must not trigger a re-provision — a flaky relay or an outage would
 * otherwise burn the (single) re-provision attempt on a key that is actually
 * fine.
 *
 * @param {number} statusCode
 * @returns {boolean}
 */
function isRelayUnauthorized(statusCode) {
  const code = Number(statusCode)
  return code === 401 || code === 403
}

// Default minimum gap between two self-heal re-provision attempts. A 401 at boot
// re-provisions once; if that fails (expired JWT, provision-key down) we must not
// retry on a tight loop (a 401 storm against the auth backend). 10 min is long
// enough that a transient failure clears by the next natural boot/probe, short
// enough that a genuine rotation heals within one session.
const REPROVISION_COOLDOWN_MS = 10 * 60 * 1000

/**
 * Decide whether a relay-401 should trigger an auto re-provision right now.
 * Pure so the gate (managed enabled + signed-in + holds a reusable login JWT)
 * and the anti-storm cooldown are unit-testable without the electron/net layer.
 *
 * Gate — ALL must hold, else 'byok'/manual is the correct outcome and we do
 * nothing (zero behavior change for BYOK / signed-out):
 *   - enabled:   managed-LLM path is on (isManagedEnabled)
 *   - hasKey:    a relay key is actually stored (a 401 only matters if we have a
 *                key that the relay rejected; no key → not our concern)
 *   - hasToken:  a login JWT is on disk to authenticate provision-key with
 *                (without it we CANNOT re-mint — the user must re-login)
 * Cooldown — even when gated in, only attempt if `cooldownMs` has elapsed since
 *   the last attempt (lastAttemptAt = 0/undefined ⇒ never tried ⇒ allowed).
 *
 * @param {{
 *   enabled?: boolean, hasKey?: boolean, hasToken?: boolean,
 *   lastAttemptAt?: number, now?: number, cooldownMs?: number
 * }} state
 * @returns {boolean}
 */
function shouldAttemptReprovision(state = {}) {
  if (!state.enabled || !state.hasKey || !state.hasToken) return false
  const now = Number.isFinite(state.now) ? state.now : Date.now()
  const last = Number.isFinite(state.lastAttemptAt) ? state.lastAttemptAt : 0
  const cooldown = Number.isFinite(state.cooldownMs) ? state.cooldownMs : REPROVISION_COOLDOWN_MS
  if (last <= 0) return true
  return now - last >= cooldown
}

/**
 * Keep the registered relay `custom_providers:` entry's api_key in lockstep
 * with the freshly provisioned relay key — pure YAML line surgery, no yaml dep.
 *
 * Why: provision-key ROTATES the key on every sign-in. The runtime's
 * /api/model/set applies the fresh key to `model.*`, but its custom-provider
 * bookkeeping (_save_custom_provider) dedups by base_url and only refreshes
 * model/api_mode — NEVER api_key. After a re-login the registered relay entry
 * therefore keeps a rotated (dead) key: the picker's live model listing 401s
 * against the relay and collapses to the single configured model.
 *
 * Matches the list entry whose base_url equals `baseUrl` (trailing slashes
 * ignored) and rewrites its api_key line when it differs from `key`. Handles
 * both the desktop-seeded shape and PyYAML's re-dump of it (`- api_key: …`
 * first line or indented, optional quotes). Anything unexpected → no change.
 *
 * @param {string} raw      config.yaml contents
 * @param {string} baseUrl  the managed relay base_url to match
 * @param {string} key      the current (fresh) relay key
 * @returns {{ changed: boolean, next: string }}
 */
function syncCustomProviderKeyYaml(raw, baseUrl, key) {
  const source = String(raw || '')
  const targetBase = String(baseUrl || '').trim().replace(/\/+$/, '')
  const freshKey = String(key || '').trim()
  if (!source || !targetBase || !freshKey) return { changed: false, next: source }

  const lines = source.split('\n')
  const unquote = value => value.trim().replace(/^(["'])(.*)\1$/, '$2')

  // Collect [start, end) line ranges of each `custom_providers:` list entry.
  const entryRanges = []
  let inList = false
  let entryStart = -1
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    if (/^custom_providers:\s*$/.test(line)) {
      inList = true
      continue
    }
    if (!inList) continue
    if (/^- /.test(line)) {
      if (entryStart >= 0) entryRanges.push([entryStart, i])
      entryStart = i
      continue
    }
    if (/^\S/.test(line)) {
      // Next top-level key — list is over.
      if (entryStart >= 0) entryRanges.push([entryStart, i])
      entryStart = -1
      inList = false
    }
  }
  if (entryStart >= 0) entryRanges.push([entryStart, lines.length])

  let changed = false
  for (const [start, end] of entryRanges) {
    let baseMatches = false
    let keyLine = -1
    for (let i = start; i < end; i++) {
      const m = lines[i].match(/^(?:- |\s+)(api_key|base_url):\s*(.*)$/)
      if (!m) continue
      if (m[1] === 'base_url' && unquote(m[2]).replace(/\/+$/, '') === targetBase) baseMatches = true
      if (m[1] === 'api_key' && keyLine < 0) keyLine = i
    }
    if (!baseMatches || keyLine < 0) continue
    const current = unquote(lines[keyLine].replace(/^.*api_key:\s*/, ''))
    if (current === freshKey) continue
    lines[keyLine] = lines[keyLine].replace(/(api_key:\s*).*$/, `$1${freshKey}`)
    changed = true
  }

  return { changed, next: changed ? lines.join('\n') : source }
}

/**
 * Extract a relay key from the (future) relay-key endpoint response, tolerating
 * the couple of shapes the backend might return. Returns null when none present
 * so the caller falls back to BYOK rather than seeding an empty key.
 *
 * @param {unknown} body
 * @returns {string | null}
 */
function relayKeyFromResponse(body) {
  if (!body || typeof body !== 'object') return null
  const candidate =
    body.relay_key ?? body.api_key ?? body.key ?? (body.item && (body.item.key ?? body.item.api_key))
  const key = typeof candidate === 'string' ? candidate.trim() : ''
  return key || null
}

/**
 * Extract a JWT access token from the login response.
 * Mirrors the cloud auth route shape: `{ access_token, token_type: 'bearer' }`.
 *
 * @param {unknown} body
 * @returns {string | null}
 */
function accessTokenFromLogin(body) {
  if (!body || typeof body !== 'object') return null
  const token = typeof body.access_token === 'string' ? body.access_token.trim() : ''
  return token || null
}

/**
 * Decode a JWT's payload (the middle base64url segment) WITHOUT verifying its
 * signature. We only read a few display claims (email / plan) for the account
 * panel — the token is never TRUSTED for authorization here (the relay validates
 * the minted key server-side; this is cosmetic identity for the signed-in user).
 * Pure + dependency-free (Buffer only), returns {} on any malformed input so a
 * weird token can never throw into the sign-in path.
 *
 * @param {string} token
 * @returns {Record<string, unknown>}
 */
function decodeJwtClaims(token) {
  const raw = String(token || '').trim()
  const parts = raw.split('.')
  if (parts.length < 2 || !parts[1]) return {}
  try {
    // base64url → base64, then decode. Buffer tolerates missing padding.
    const b64 = parts[1].replace(/-/g, '+').replace(/_/g, '/')
    const json = Buffer.from(b64, 'base64').toString('utf8')
    const claims = JSON.parse(json)
    return claims && typeof claims === 'object' ? claims : {}
  } catch {
    return {}
  }
}

/**
 * Build the display-only account descriptor persisted alongside the relay key so
 * the desktop account panel can show who is signed in. Reads the email + plan
 * from the login/register response body first, then falls back to the JWT
 * claims (the cloud auth route encodes `email`/`sub` and may include a
 * plan/tier). Everything is optional — a missing field just renders a generic
 * label. NEVER includes the token or any secret; this object is stored in clear.
 *
 * @param {unknown} loginBody   response of /auth/login or /auth/register
 * @param {string} [accessToken] the JWT (claims used as a fallback source)
 * @returns {{ email: string, name: string, plan: string }}
 */
function accountFromLogin(loginBody, accessToken = '') {
  const body = loginBody && typeof loginBody === 'object' ? loginBody : {}
  const claims = decodeJwtClaims(accessToken)
  const str = value => (typeof value === 'string' ? value.trim() : '')
  // Tolerate the couple of shapes the backend / JWT might use for each field.
  const email = str(body.email) || str(claims.email) || str(claims.sub && String(claims.sub).includes('@') ? claims.sub : '')
  const name = str(body.name) || str(body.display_name) || str(claims.name)
  const plan = str(body.plan) || str(body.tier) || str(claims.plan) || str(claims.tier)
  return { email, name, plan }
}

module.exports = {
  DEFAULT_AUTH_BASE,
  DEFAULT_API_BASE,
  DEFAULT_RELAY_BASE_URL,
  DEFAULT_MANAGED_MODEL,
  MANAGED_MODEL_DISPLAY,
  MANAGED_PROVIDER,
  MANAGED_PROVIDER_NAME,
  MODEL_DISABLED_PROVIDERS,
  SEED_DISABLED_SKILLS,
  MANAGED_PLUGIN_NAMES,
  REPROVISION_COOLDOWN_MS,
  modelDisabledProvidersYaml,
  seedSkillsBlockYaml,
  seedPluginsBlockYaml,
  ensurePluginsEnabledYaml,
  ensureSkillsDisabledYaml,
  LOGIN_PATH,
  REGISTER_PATH,
  PROVISION_KEY_PATH,
  GOOGLE_START_PATH,
  WEB_LOGIN_PATH,
  accessTokenFromLogin,
  accountFromLogin,
  apexWebLoginUrl,
  buildManagedModelConfig,
  decodeJwtClaims,
  defaultModelPath,
  googleStartUrl,
  isLoopbackUrl,
  isManagedEnabled,
  isRelayUnauthorized,
  managedModelConfigYaml,
  parseLoopbackCallback,
  parseProvisionResponse,
  relayKeyFromResponse,
  resolveApexEndpoints,
  shouldAttemptReprovision,
  syncCustomProviderKeyYaml
}
