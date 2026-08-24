import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// Edit Profile always sends the avatar image in its saveBotMeta patch
// (changed or not), and saveBotMeta fired profiles.set_asset for every patch
// carrying the key. Each save re-uploaded the full data URL or fired a
// `clear` — and a no-op `clear` from one machine could race another
// machine's just-pushed avatar and wipe it server-side. set_asset now fires
// only when the image actually changed; ui_meta sync is untouched.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function load() {
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

test('regression: set_asset fires only when the avatar image changes', async () => {
  const { saveBotMeta, $botMeta, requests } = load()
  const png = 'data:image/png;base64,AAAA'

  saveBotMeta('ops', { image: png, title: 'One' })
  saveBotMeta('ops', { image: png, title: 'Two' }) // re-save, same image
  saveBotMeta('ops', { image: null, title: 'Three' }) // removed
  saveBotMeta('ops', { title: 'Four' }) // no image key at all
  await Promise.resolve()

  const assetCalls = JSON.parse(
    JSON.stringify(requests.filter(([method]) => method === 'profiles.set_asset').map(([, params]) => params))
  )
  assert.deepEqual(assetCalls, [
    { name: 'ops', asset: 'avatar', data: png },
    { name: 'ops', asset: 'avatar', clear: true }
  ])

  // Meta still merges per patch, and ui_meta sync is unaffected by the
  // image-gating.
  assert.equal($botMeta.get().ops.title, 'Four')
  assert.equal(requests.filter(([method]) => method === 'profiles.configure').length, 4)
})

test('regression: duplicating a bot still pushes the copied avatar once', async () => {
  const { saveBotMeta, requests } = load()
  const png = 'data:image/png;base64,BBBB'

  saveBotMeta('source', { image: png, title: 'Original' })
  saveBotMeta('source-2', { image: png, title: 'Original (copy)' })
  await Promise.resolve()

  const assetCalls = JSON.parse(
    JSON.stringify(requests.filter(([method]) => method === 'profiles.set_asset').map(([, params]) => params))
  )
  assert.deepEqual(assetCalls, [
    { name: 'source', asset: 'avatar', data: png },
    { name: 'source-2', asset: 'avatar', data: png } // fresh profile: image differs from its (empty) meta
  ])
})
