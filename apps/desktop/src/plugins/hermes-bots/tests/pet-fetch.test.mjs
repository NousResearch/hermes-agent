import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function load(fetchImpl) {
  const values = new Map()
  const atom = initial => {
    const slot = { get: () => values.get(slot), set: value => values.set(slot, value) }
    values.set(slot, initial)
    return slot
  }
  const fetches = []
  const context = {
    atom,
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    document: {
      getElementById: () => null,
      createElement: () => ({
        width: 0,
        height: 0,
        getContext: () => ({ drawImage() {} }),
        toDataURL: () => 'data:image/png;base64,ok'
      }),
      head: { appendChild: () => undefined }
    },
    host: { state: { profile: { listen: () => undefined } } },
    fetch: async (url, opts) => {
      fetches.push({ url, opts })
      return fetchImpl(url, opts)
    },
    AbortSignal,
    createImageBitmap: async () => ({ close() {} })
  }
  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat('\nglobalThis.__api = { petFrameIcon };\n')
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  return { context, fetches }
}

test('unit: a failed pet fetch is not stuck in the cache', async () => {
  let n = 0
  const { context, fetches } = load(async () => {
    n += 1
    throw new Error('network')
  })
  const a = await context.__api.petFrameIcon('https://pets.example/a.webp')
  const b = await context.__api.petFrameIcon('https://pets.example/a.webp')
  assert.equal(a, null)
  assert.equal(b, null)
  assert.equal(fetches.length, 2)
  assert.equal(n, 2)
  assert.ok(fetches[0].opts.signal, 'fetch is aborted if it hangs')
})

test('regression: a successful icon is still cached', async () => {
  let n = 0
  const { context } = load(async () => {
    n += 1
    return { blob: async () => new Blob() }
  })
  const a = await context.__api.petFrameIcon('https://pets.example/ok.webp')
  const b = await context.__api.petFrameIcon('https://pets.example/ok.webp')
  assert.equal(a, 'data:image/png;base64,ok')
  assert.equal(n, 1)
  assert.equal(a, b)
})
