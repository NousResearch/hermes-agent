import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

/** Load the plugin in a sandbox. `request` stubs host.request; `fetch` stubs
 *  the (now unused for pet thumbs) global fetch. Returns the context plus the
 *  recorded host.request calls. */
function load({ request, fetch } = {}) {
  const values = new Map()
  const atom = initial => {
    const slot = { get: () => values.get(slot), set: value => values.set(slot, value) }
    values.set(slot, initial)
    return slot
  }
  const requests = []
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
    host: {
      state: { profile: { listen: () => undefined } },
      request: async (method, params) => {
        requests.push({ method, params })
        return request ? request(method, params) : { ok: true, dataUri: 'data:image/png;base64,ok' }
      }
    },
    fetch: async (url, opts) => {
      fetches.push({ url, opts })
      return fetch ? fetch(url, opts) : { blob: async () => new Blob() }
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
    .concat('\nglobalThis.__api = { petThumbIcon };\n')
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  return { context, requests, fetches }
}

test('regression: a local-only pet (no spritesheet URL) still resolves its icon via pet.thumb', async () => {
  // Local-only pet: generator-hatched, absent from the petdex manifest, so
  // pet.gallery hands the picker an EMPTY spritesheetUrl. The old client-side
  // fetch path bailed on that and the picker refused the selection. pet.thumb
  // reads the installed sheet off disk server-side, so it must still work.
  let n = 0
  const { context, requests } = load({
    request: async (method, params) => {
      n += 1
      assert.equal(method, 'pet.thumb')
      assert.equal(params.slug, 'local-pet')
      assert.equal(params.url, '')
      return { ok: true, slug: 'local-pet', dataUri: 'data:image/png;base64,local-pet-thumb' }
    }
  })
  const icon = await context.__api.petThumbIcon('local-pet', '')
  assert.equal(icon, 'data:image/png;base64,local-pet-thumb')
  assert.equal(n, 1)
  assert.equal(requests.length, 1)
  assert.equal(requests[0].method, 'pet.thumb')
})

test('unit: a failed pet thumb is not stuck in the cache', async () => {
  let n = 0
  const { context, requests } = load({
    request: async () => {
      n += 1
      throw new Error('backend unavailable')
    }
  })
  const a = await context.__api.petThumbIcon('local-pet', '')
  const b = await context.__api.petThumbIcon('local-pet', '')
  assert.equal(a, null)
  assert.equal(b, null)
  assert.equal(requests.length, 2)
  assert.equal(n, 2)
})

test('unit: an ok:false thumb (no usable source) resolves null and is retried', async () => {
  let n = 0
  const { context } = load({
    request: async () => {
      n += 1
      return { ok: false, slug: 'local-pet' }
    }
  })
  assert.equal(await context.__api.petThumbIcon('local-pet', ''), null)
  assert.equal(await context.__api.petThumbIcon('local-pet', ''), null)
  assert.equal(n, 2)
})

test('regression: a successful icon is still cached', async () => {
  let n = 0
  const { context, requests } = load({
    request: async () => {
      n += 1
      return { ok: true, slug: 'dogalright', dataUri: 'data:image/png;base64,ok' }
    }
  })
  const a = await context.__api.petThumbIcon('dogalright', 'https://assets.petdex.dev/.../dogalright.webp')
  const b = await context.__api.petThumbIcon('dogalright', 'https://assets.petdex.dev/.../dogalright.webp')
  assert.equal(a, 'data:image/png;base64,ok')
  assert.equal(b, a)
  assert.equal(requests.length, 1)
  assert.equal(n, 1)
})
