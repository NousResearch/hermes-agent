import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// #95293 remainder: a GUARDED model switch made from the Bots surface must
// surface the SAME confirm flow the core picker uses, then resend with
// `confirm_expensive_model: true` once the user confirms.
//
// The Bots editor's switch path is `applyAdvancedConfig` → `profiles.configure`
// (it does NOT route through use-model-controls). The gateway now answers
// `confirm_required` + `confirm_message` for guarded picks on this surface too
// (same handshake as `config.set model`); the plugin must:
//
//   1. route that response through the shared confirm handler
//      (`surfaceModelSwitchConfirm` from the plugin SDK — one applier, no
//      forked confirm logic per surface),
//   2. NOT count the pending model section as a failed section, and
//   3. resend ONLY the model section with `confirm_expensive_model: true`
//      when the user clicks Confirm.
//
// Pre-fix, the confirm_required response matched no branch: the pick silently
// dropped (no confirm, no resend) — exactly issue #95293 on the Bots surface.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadHarness({ request }) {
  const values = new Map()
  const atom = initial => {
    const slot = {
      get: () => values.get(slot),
      set: value => values.set(slot, value),
      listen: () => undefined
    }
    values.set(slot, initial)
    return slot
  }
  const jsx = (type, props = {}) => ({ type, props })
  const routed = []
  const confirms = []
  const invalidated = []
  const context = {
    atom,
    Checkbox: 'Checkbox',
    GlyphSpinner: 'GlyphSpinner',
    Input: 'Input',
    ScrollArea: 'ScrollArea',
    Textarea: 'Textarea',
    document: { createElement: () => ({}), getElementById: () => null, head: { appendChild: () => undefined } },
    host: {
      getGateway: () => 'ambient-gateway',
      request: async (method, params) => {
        routed.push([method, params])

        return request(method, params)
      },
      requestProfile: async (route, method, params) => {
        routed.push([method, params])

        return request(method, params)
      },
      state: {
        gateway: { get: () => 'open', listen: () => undefined },
        profile: { get: () => 'default', listen: () => undefined }
      }
    },
    jsx,
    jsxs: jsx,
    queryClient: { invalidateQueries: key => invalidated.push(key) },
    sdk: {},
    // The shared confirm handler the core picker uses, exported through the
    // plugin SDK. The stub records the options so the test can drive the
    // Confirm action exactly like a user click would.
    surfaceModelSwitchConfirm: options => {
      confirms.push(options)

      return 'notification-1'
    },
    useQuery: () => ({ data: undefined, isLoading: false, error: null }),
    useState: initial => [initial, () => undefined],
    window: { setTimeout, clearTimeout }
  }
  const code = source
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat('\nglobalThis.__applyAdvancedConfig = applyAdvancedConfig;')
  vm.runInNewContext(code, context, { filename: 'plugin.js' })

  return { applyAdvancedConfig: context.__applyAdvancedConfig, confirms, invalidated, routed }
}

const bot = { name: 'zeta' }

const dirtyModelState = {
  loaded: true,
  provider: 'opencode-go',
  model: 'muse-spark-1.2-contributor',
  soul: '',
  skills: [],
  toolsets: [],
  mcp: [],
  dirtyModel: true,
  dirtySoul: false,
  dirtySkills: false,
  dirtyToolsets: false,
  dirtyMcp: false
}

test('guarded Bots-mode switch surfaces the shared confirm flow instead of silently dropping', async () => {
  const { applyAdvancedConfig, confirms, routed } = loadHarness({
    request: async () => ({
      ok: true,
      applied: {},
      confirm_required: true,
      confirm_message: 'CONTRIBUTOR TIER: this model may train on your data.'
    })
  })

  const res = await applyAdvancedConfig(bot, { ...dirtyModelState })

  // The first request is the normal (unconfirmed) configure call.
  assert.equal(routed.length, 1)
  assert.equal(routed[0][0], 'profiles.configure')
  assert.equal(routed[0][1].model, 'muse-spark-1.2-contributor')
  assert.ok(!routed[0][1].confirm_expensive_model, 'first attempt must NOT pre-confirm')

  // The confirm flow was surfaced through the SHARED handler with the
  // gateway's message — not swallowed, not a forked local dialog.
  assert.equal(confirms.length, 1, 'confirm_required must surface the shared confirm handler')
  assert.match(String(confirms[0].confirmMessage || ''), /CONTRIBUTOR TIER/)

  // Pending-confirmation is NOT a failed section: the editor must not toast
  // "Some sections failed: model" while the confirm toast is up.
  assert.notEqual(res?.applied?.model, false)
  assert.equal(res?.ok, true)
})

test('Confirm resends ONLY the model section with confirm_expensive_model: true', async () => {
  let calls = 0
  const { applyAdvancedConfig, confirms, routed } = loadHarness({
    request: async () => {
      calls += 1

      if (calls === 1) {
        return { ok: true, applied: {}, confirm_required: true, confirm_message: 'guarded' }
      }

      return { ok: true, applied: { model: true } }
    }
  })

  await applyAdvancedConfig(bot, { ...dirtyModelState })
  assert.equal(confirms.length, 1)

  // Drive the confirmed resend exactly as the shared handler's Confirm
  // action does.
  const confirmed = await confirms[0].requestConfirmed()

  assert.equal(routed.length, 2)
  const [method, params] = routed[1]
  assert.equal(method, 'profiles.configure')
  assert.equal(params.confirm_expensive_model, true)
  assert.equal(params.model, 'muse-spark-1.2-contributor')
  assert.equal(params.provider, 'opencode-go')
  assert.equal(params.name, 'zeta')
  // Only the model section rides the resend — no soul/skills/toolsets replay.
  assert.deepEqual(
    Object.keys(params).sort(),
    ['confirm_expensive_model', 'model', 'name', 'provider']
  )
  assert.equal(confirmed?.applied?.model, true)
})

test('unguarded saves keep the existing single-shot behavior', async () => {
  const { applyAdvancedConfig, confirms, routed } = loadHarness({
    request: async () => ({ ok: true, applied: { model: true } })
  })

  const res = await applyAdvancedConfig(bot, { ...dirtyModelState })

  assert.equal(routed.length, 1)
  assert.equal(confirms.length, 0)
  assert.equal(res?.ok, true)
  assert.equal(res?.applied?.model, true)
})
