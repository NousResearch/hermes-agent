import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadApplyAdvancedConfig(request) {
  const start = source.indexOf('async function applyAdvancedConfig(')
  const end = source.indexOf('// ── edit profile dialog', start)
  const context = {
    host: { request },
    requestForBot: (_bot, method, params) => request(method, params),
    ensureMessagingProtocol: soul => soul,
    $lastRoster: { get: () => [] }
  }
  const section = source
    .slice(start, end)
    .concat('\nglobalThis.__advanced = { applyAdvancedConfig };\n')

  assert.notEqual(start, -1, 'applyAdvancedConfig is missing')
  assert.notEqual(end, -1, 'advanced-config section delimiter is missing')
  vm.runInNewContext(section, context, { filename: 'advanced-config.js' })
  return context.__advanced.applyAdvancedConfig
}

function state(patch = {}) {
  return {
    provider: '',
    model: '',
    soul: '',
    skills: [],
    toolsets: [],
    dirtyModel: false,
    dirtySoul: false,
    dirtySkills: false,
    dirtyToolsets: false,
    ...patch
  }
}

test('regression: selecting Inherit explicitly clears the profile model assignment', async () => {
  const calls = []
  const apply = loadApplyAdvancedConfig(async (method, params) => {
    calls.push({ method, params })
    return { code: 0, blocked: false, output: 'Unset model' }
  })

  const result = await apply({ name: 'ops' }, state({ dirtyModel: true }))

  assert.deepEqual(JSON.parse(JSON.stringify(calls)), [
    {
      method: 'cli.exec',
      params: { argv: ['--profile', 'ops', 'config', 'unset', 'model'] }
    }
  ])
  assert.deepEqual(JSON.parse(JSON.stringify(result)), { ok: true, applied: { model: true } })
})

test('integration: model clearing and other dirty sections report a merged result', async () => {
  const calls = []
  const apply = loadApplyAdvancedConfig(async (method, params) => {
    calls.push({ method, params })
    if (method === 'cli.exec') return { code: 0, blocked: false, output: 'Unset model' }
    return { ok: true, applied: { soul: true } }
  })

  const result = await apply({ name: 'ops' }, state({ dirtyModel: true, dirtySoul: true, soul: '# Ops' }))

  assert.equal(calls[0].method, 'cli.exec')
  assert.deepEqual(JSON.parse(JSON.stringify(calls[1])), {
    method: 'profiles.configure',
    params: { name: 'ops', soul: '# Ops' }
  })
  assert.deepEqual(JSON.parse(JSON.stringify(result)), {
    ok: true,
    applied: { model: true, soul: true }
  })
})

test('regression: a rejected model clear is reported as a failed section', async () => {
  const apply = loadApplyAdvancedConfig(async () => ({ code: 1, blocked: false, output: 'Config key not set' }))

  const result = await apply({ name: 'ops' }, state({ dirtyModel: true }))

  assert.deepEqual(JSON.parse(JSON.stringify(result)), { ok: false, applied: { model: false } })
})

test('regression: an advanced-section failure suppresses the contradictory success toast', () => {
  const start = source.indexOf('function EditProfileDialog(')
  const end = source.indexOf('function CreateAgentDialog(', start)
  const dialog = source.slice(start, end)

  assert.match(dialog, /let advancedFailed = false/)
  assert.match(dialog, /if \(!advancedFailed && !lookFailed\) \{\s*host\.notify\(\{ kind: 'success'/)
})
