import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function load(storageGet) {
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
    host: {
      request: async () => ({}),
      state: {
        profile: { listen: () => undefined },
        gateway: { listen: () => undefined }
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
    .concat('\nglobalThis.__api = { saveBotMeta, $botMeta, plugin };\n')
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  context.plugin.register({
    storage: { get: storageGet, set: () => undefined },
    register: () => undefined
  })
  return context.__api
}

test('regression: a slow load does not wipe a pin written this session', async () => {
  let resolveStorage
  const pending = new Promise(resolve => {
    resolveStorage = resolve
  })
  const runtime = load(() => pending)

  runtime.saveBotMeta('researcher', { chat: 'sess-just-created', title: 'Research' })
  assert.equal(runtime.$botMeta.get().researcher.chat, 'sess-just-created')

  resolveStorage({ researcher: { title: 'Research' } })
  await pending
  await Promise.resolve()
  await Promise.resolve()

  assert.equal(runtime.$botMeta.get().researcher.chat, 'sess-just-created')
  assert.equal(runtime.$botMeta.get().researcher.title, 'Research')
})

test('regression: a slow load still keeps looks from disk under a new pin', async () => {
  let resolveStorage
  const pending = new Promise(resolve => {
    resolveStorage = resolve
  })
  const runtime = load(() => pending)

  runtime.saveBotMeta('researcher', { chat: 'sess-just-created' })
  resolveStorage({ researcher: { title: 'Research', shape: 'cloud', color: '#38bdf8' } })
  await pending
  await new Promise(resolve => setTimeout(resolve, 0))

  assert.equal(runtime.$botMeta.get().researcher.chat, 'sess-just-created')
  assert.equal(runtime.$botMeta.get().researcher.shape, 'cloud')
  assert.equal(runtime.$botMeta.get().researcher.title, 'Research')
})

test('unit: disk still fills in bots we have not touched yet', async () => {
  const runtime = load(() => ({
    researcher: { title: 'Research', chat: 'sess-old' },
    writer: { title: 'Writer' }
  }))

  await Promise.resolve()
  await Promise.resolve()

  assert.equal(runtime.$botMeta.get().researcher.chat, 'sess-old')
  assert.equal(runtime.$botMeta.get().writer.title, 'Writer')
})
