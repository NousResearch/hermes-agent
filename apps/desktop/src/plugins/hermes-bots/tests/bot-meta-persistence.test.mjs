import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function load(configureResponse) {
  const values = new Map()
  const atom = initial => {
    const slot = { get: () => values.get(slot), set: value => values.set(slot, value) }
    values.set(slot, initial)
    return slot
  }
  const requests = []
  const context = {
    atom,
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: {
      request: (method, params) => {
        requests.push([method, params])
        if (method === 'profiles.configure') {
          return typeof configureResponse === 'function' ? configureResponse() : Promise.resolve(configureResponse)
        }
        return Promise.resolve({})
      },
      state: { profile: { get: () => 'default', listen: () => undefined }, gateway: { listen: () => undefined } }
    }
  }
  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat('\nglobalThis.__meta = { saveBotMeta, $botMeta };\n')
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  context.plugin.register({
    storage: { get: () => null, set: () => undefined },
    register: () => undefined
  })
  return { ...context.__meta, requests }
}

test('server persistence failure is returned while the local look remains saved', async () => {
  const { saveBotMeta, $botMeta } = load({ applied: { ui_meta: false } })

  const result = await saveBotMeta('researcher', {
    shape: 'cloud',
    color: '#38bdf8',
    custom: true
  })

  assert.equal($botMeta.get().researcher.shape, 'cloud')
  assert.equal($botMeta.get().researcher.color, '#38bdf8')
  assert.equal(result.serverPersisted, false)
  assert.equal(result.serverOutcome, 'failed')
})

test('server persistence success is returned after ui_meta is applied', async () => {
  const { saveBotMeta } = load({ applied: { ui_meta: true } })

  const result = await saveBotMeta('researcher', {
    shape: 'hexagon',
    color: '#8b5cf6',
    custom: true
  })

  assert.equal(result.serverPersisted, true)
  assert.equal(result.serverOutcome, 'persisted')
})

test('an older gateway (no applied contract) is unsupported, not failed', async () => {
  const { saveBotMeta } = load({})

  const result = await saveBotMeta('researcher', { shape: 'circle', custom: true })

  assert.equal(result.serverOutcome, 'unsupported')
  assert.equal(result.serverPersisted, false)
})

test('a rejected configure request is unsupported, not failed', async () => {
  const { saveBotMeta } = load(() => Promise.reject(new Error('param shape rejected')))

  const result = await saveBotMeta('researcher', { shape: 'circle', custom: true })

  assert.equal(result.serverOutcome, 'unsupported')
})

test('Edit Profile waits for persistence and reports a local-only save', () => {
  assert.match(pluginSource, /const persistence = await saveBotMeta\(bot,/)
  assert.match(pluginSource, /Saved look locally; remote persistence failed/)
  // The toast fires only on an explicit remote failure — never on the
  // documented older-gateway fallback.
  assert.match(pluginSource, /serverOutcome === 'failed'/)
})
