import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function load() {
  const values = new Map()
  const atom = initial => {
    const slot = { get: () => values.get(slot), set: value => values.set(slot, value) }
    values.set(slot, initial)
    return slot
  }
  const context = {
    atom,
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: { state: { profile: { listen: () => undefined } } },
    sdk: new Proxy({}, { get: () => undefined })
  }
  const source = pluginSource
    .replace(/^import \* as sdk from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat(`
globalThis.__botMetaSync = {
  get: () => $botMeta.get(),
  set: value => $botMeta.set(value),
  mergeServerMeta,
  setPluginCtx: value => { pluginCtx = value }
};
`)
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  return context.__botMetaSync
}

test('regression: authoritative server metadata removes stale local canonical chat', () => {
  const writes = []
  const sync = load()
  sync.set({
    default: {
      chat: '4f6f6798ad6f',
      shape: 'cloud',
      image: 'data:image/png;base64,avatar',
      pet: 'pet-cat'
    }
  })
  sync.setPluginCtx({ storage: { set: (key, value) => writes.push({ key, value }) } })

  sync.mergeServerMeta([
    {
      name: 'default',
      ui_meta: {
        'hermes-bots': { shape: 'cloud', color: '#8b5cf6', title: 'Hermana' }
      }
    }
  ])

  const current = sync.get().default
  assert.equal(Object.hasOwn(current, 'chat'), false)
  assert.equal(current.image, 'data:image/png;base64,avatar')
  assert.equal(current.pet, 'pet-cat')
  assert.equal(writes.at(-1).key, 'bot-meta')
  assert.equal(Object.hasOwn(writes.at(-1).value.default, 'chat'), false)
})

test('regression: authoritative groups remove a stale local legacy group projection', () => {
  const writes = []
  const sync = load()
  sync.set({ researcher: { groups: ['Old'], group: 'Old', title: 'Researcher' } })
  sync.setPluginCtx({ storage: { set: (key, value) => writes.push({ key, value }) } })

  sync.mergeServerMeta([
    {
      name: 'researcher',
      ui_meta: { 'hermes-bots': { groups: [], title: 'Researcher' } }
    }
  ])

  const current = sync.get().researcher
  assert.deepEqual(current.groups, [])
  assert.equal(Object.hasOwn(current, 'group'), false)
  assert.equal(Object.hasOwn(writes.at(-1).value.researcher, 'group'), false)
})

test('compatibility: local canonical chat survives when gateway has no server bot metadata', () => {
  const writes = []
  const sync = load()
  sync.set({ default: { chat: 'local-chat', shape: 'cloud' } })
  sync.setPluginCtx({ storage: { set: (key, value) => writes.push({ key, value }) } })

  sync.mergeServerMeta([{ name: 'default' }])

  assert.equal(sync.get().default.chat, 'local-chat')
  assert.equal(writes.length, 0)
})
