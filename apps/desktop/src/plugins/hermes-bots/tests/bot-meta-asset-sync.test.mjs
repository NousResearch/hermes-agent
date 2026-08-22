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
  const profileRequests = []
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
      requestProfile: (route, method, params) => {
        profileRequests.push([route, method, params])
        return Promise.resolve(
          method === 'profiles.get_asset'
            ? { found: true, data: 'data:image/png;base64,REMOTE' }
            : {}
        )
      },
      state: {
        profile: { get: () => 'default', listen: () => undefined },
        connectionId: { get: () => 'local', listen: () => undefined },
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
    .concat(
      '\nglobalThis.__meta = { saveBotMeta, $botMeta, $remoteBotMeta, pullServerAvatars, botRosterMeta };\n'
    )
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  context.plugin.register({
    storage: { get: () => null, set: () => undefined },
    register: () => undefined
  })
  return { ...context.__meta, requests, profileRequests }
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

test('remote avatars are fetched through their source and cached by source-qualified key', async () => {
  const { pullServerAvatars, botRosterMeta, $botMeta, $remoteBotMeta, requests, profileRequests } = load()
  const remote = {
    name: 'default',
    connectionId: 'homelab',
    remoteSource: true,
    has_avatar: true,
    ui_meta: { 'hermes-bots': { title: 'Homelab Hermes' } }
  }

  pullServerAvatars([remote])
  await new Promise(resolve => setImmediate(resolve))

  assert.equal(requests.filter(([method]) => method === 'profiles.get_asset').length, 0)
  assert.equal(profileRequests.length, 1)
  assert.deepEqual(JSON.parse(JSON.stringify(profileRequests[0])), [
    {
      connectionId: 'homelab',
      mode: 'remote',
      profile: 'default',
      targetProfile: 'default'
    },
    'profiles.get_asset',
    { name: 'default', asset: 'avatar' }
  ])
  assert.equal($botMeta.get().default, undefined)
  assert.equal($remoteBotMeta.get()['homelab::default'].image, 'data:image/png;base64,REMOTE')
  assert.equal(botRosterMeta(remote, $botMeta.get()).title, 'Homelab Hermes')
  assert.equal(botRosterMeta(remote, $botMeta.get()).image, 'data:image/png;base64,REMOTE')
})
