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
  const calls = []
  const context = {
    atom,
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: {
      state: { profile: { listen: () => undefined } },
      request: async (method, params) => {
        calls.push({ method, params })
        return { ok: true }
      }
    }
  }
  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat('\nglobalThis.__dup = { duplicateBot, $botMeta };\n')
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  return { context, calls }
}

test('unit: duplicate copies the look but not the source chat pin', async () => {
  const { context, calls } = load()
  const { duplicateBot, $botMeta } = context.__dup
  $botMeta.set({
    researcher: {
      shape: 'circle',
      color: '#f97316',
      title: 'Researcher',
      image: 'data:image/png;base64,xx',
      chat: 'sess-source-forever',
      created: 1700000000000
    }
  })

  const name = await duplicateBot({ name: 'researcher', description: 'finds things' }, [{ name: 'researcher' }])
  assert.equal(name, 'researcher-2')

  const clone = $botMeta.get()['researcher-2']
  assert.equal(clone.shape, 'circle')
  assert.equal(clone.color, '#f97316')
  assert.equal(clone.image, 'data:image/png;base64,xx')
  assert.equal(clone.title, 'Researcher (copy)')
  assert.equal(clone.chat, undefined)
  assert.equal(clone.created, undefined)

  const created = calls.find(c => c.method === 'profiles.create')
  assert.equal(created.params.clone_from, 'researcher')
  assert.equal(created.params.name, 'researcher-2')

  const configure = calls.filter(c => c.method === 'profiles.configure')
  assert.ok(configure.length)
  const ui = configure[configure.length - 1].params.ui_meta['hermes-bots']
  assert.equal(ui.chat, undefined)
  assert.equal(ui.created, undefined)
  assert.equal(ui.title, 'Researcher (copy)')
})

test('regression: a bot with no pin still duplicates its look', async () => {
  const { context } = load()
  const { duplicateBot, $botMeta } = context.__dup
  $botMeta.set({ painter: { shape: 'cloud', color: '#38bdf8', title: 'Painter' } })
  const name = await duplicateBot({ name: 'painter' }, [{ name: 'painter' }])
  const clone = $botMeta.get()[name]
  assert.equal(clone.title, 'Painter (copy)')
  assert.equal(clone.shape, 'cloud')
  assert.equal(clone.chat, undefined)
})
