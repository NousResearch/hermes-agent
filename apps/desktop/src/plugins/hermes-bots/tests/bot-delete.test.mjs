import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function load({ activeConnectionId = 'local', activeProfile = 'default', sdkDeleteProfile = false } = {}) {
  const requests = []
  const sdkDeletes = []
  const invalidations = []
  const newChats = []
  const stored = []
  const storageOps = []
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
      state: {
        profile: { get: () => activeProfile, listen: () => undefined },
        connectionId: { get: () => activeConnectionId, listen: () => undefined },
        gateway: { listen: () => undefined }
      },
      newChat: profile => newChats.push(profile),
      ...(sdkDeleteProfile
        ? {
            deleteProfile: async profile => {
              sdkDeletes.push(profile)
            }
          }
        : {}),
      request: async (method, params) => {
        requests.push({ method, params })
        return method === 'cli.exec' ? { blocked: false, code: 0, output: 'deleted' } : { ok: true }
      }
    },
    queryClient: { invalidateQueries: value => invalidations.push(value) }
  }
  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat('\nglobalThis.__delete = { deleteBot, $botMeta, $botUnread, $selectedBot };\n')
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  context.plugin.register({
    storage: {
      get: () => null,
      remove: key => storageOps.push({ op: 'remove', key }),
      set: (key, value) => {
        stored.push({ key, value })
        storageOps.push({ op: 'set', key, value })
      }
    },
    register: () => undefined
  })
  return { context, requests, sdkDeletes, invalidations, newChats, stored, storageOps }
}

test('unit: deleting a bot prefers the SDK deleteProfile (teardown-routed REST path)', async () => {
  const { context, requests, sdkDeletes } = load({ sdkDeleteProfile: true })

  await context.__delete.deleteBot({ name: 'researcher' })

  assert.deepEqual(sdkDeletes, ['researcher'])
  assert.equal(requests.filter(r => r.method === 'cli.exec').length, 0)
})

test('unit: source-scoped deletion preserves the route targetProfile', async () => {
  const { context, sdkDeletes } = load({ sdkDeleteProfile: true })
  const route = {
    connectionId: 'source-a',
    mode: 'remote',
    profile: 'worker',
    targetProfile: 'backend-worker'
  }

  await context.__delete.deleteBot({ name: 'worker', sourceScoped: true, route })

  assert.deepEqual(JSON.parse(JSON.stringify(sdkDeletes)), [route])
})

test('unit: deleting same-name bot on another connection keeps the active chat', async () => {
  const { context, newChats } = load({
    activeConnectionId: 'source-b',
    activeProfile: 'worker',
    sdkDeleteProfile: true
  })

  await context.__delete.deleteBot({
    name: 'worker',
    sourceScoped: true,
    route: {
      connectionId: 'source-a',
      mode: 'remote',
      profile: 'worker',
      targetProfile: 'worker'
    }
  })

  assert.deepEqual(newChats, [])
})

test('unit: source-scoped deletion commits bot-meta-v2 atomically', async () => {
  const { context, storageOps } = load({ sdkDeleteProfile: true })
  const route = {
    connectionId: 'source-a',
    mode: 'remote',
    profile: 'worker',
    targetProfile: 'backend-worker'
  }
  const key = 'source-a::worker'
  context.__delete.$botMeta.set({
    [key]: { title: 'Worker' },
    'source-a::writer': { title: 'Writer' }
  })

  await context.__delete.deleteBot({
    name: 'worker',
    sourceScoped: true,
    route
  })

  assert.deepEqual(
    JSON.parse(JSON.stringify(storageOps.slice(-3))),
    [
      { op: 'remove', key: 'bot-meta-v2-migrated' },
      {
        op: 'set',
        key: 'bot-meta-v2',
        value: { 'source-a::writer': { title: 'Writer' } }
      },
      { op: 'set', key: 'bot-meta-v2-migrated', value: true }
    ]
  )
})

test('unit: an alias targeting backend default is rejected before deletion', async () => {
  const { context, sdkDeletes, requests } = load({ sdkDeleteProfile: true })

  await assert.rejects(
    context.__delete.deleteBot({
      name: 'worker',
      sourceScoped: true,
      route: {
        connectionId: 'source-a',
        mode: 'remote',
        profile: 'worker',
        targetProfile: 'default'
      }
    }),
    /default profile cannot be deleted/i
  )

  assert.deepEqual(sdkDeletes, [])
  assert.equal(requests.some(request => request.method === 'cli.exec'), false)
})

test('unit: source-scoped deletion without host.deleteProfile fails closed without CLI execution', async () => {
  const { context, requests } = load()

  await assert.rejects(
    context.__delete.deleteBot({
      name: 'worker',
      sourceScoped: true,
      route: {
        connectionId: 'source-a',
        mode: 'remote',
        profile: 'worker',
        targetProfile: 'worker'
      }
    }),
    /source-scoped profile deletion requires host\.deleteProfile/i
  )

  assert.equal(requests.some(request => request.method === 'cli.exec'), false)
})

test('unit: older desktop without host.deleteProfile falls back to the non-interactive profile CLI command', async () => {
  const { context, requests } = load()

  await context.__delete.deleteBot({ name: 'researcher' })

  // The load-time hide sweep (hideOwnedBotSessions/sweepBotProfileSessions)
  // may interleave its own profiles.list / session.* traffic — filter to the
  // delete path's request.
  const exec = requests.filter(r => r.method === 'cli.exec')
  assert.deepEqual(JSON.parse(JSON.stringify(exec[0])), {
    method: 'cli.exec',
    params: { argv: ['profile', 'delete', 'researcher', '--yes'] }
  })
})

test('integration: a deleted bot is removed from plugin-local state and the roster is refreshed', async () => {
  const { context, invalidations, stored } = load()
  context.__delete.$botMeta.set({ researcher: { title: 'Research' }, writer: { title: 'Writer' } })
  context.__delete.$botUnread.set({ researcher: true, writer: true })
  context.__delete.$selectedBot.set('researcher')

  await context.__delete.deleteBot({ name: 'researcher' })

  assert.equal(context.__delete.$botMeta.get().researcher, undefined)
  assert.equal(context.__delete.$botUnread.get().researcher, undefined)
  assert.equal(context.__delete.$selectedBot.get(), 'default')
  assert.equal(stored.at(-1).key, 'bot-meta')
  assert.equal(stored.at(-1).value.researcher, undefined)
  assert.equal(invalidations.length, 1)
})

test('regression: the bot context menu exposes a destructive delete action and confirmation', () => {
  assert.match(pluginSource, /ContextMenuItem, \{[\s\S]*?variant: 'destructive'[\s\S]*?children: 'Delete'/)
  assert.match(pluginSource, /isDefaultBot\(bot\) \? null : jsx\(ContextMenuSeparator/)
  assert.match(pluginSource, /ConfirmDialog/)
  assert.match(pluginSource, /title: 'Delete bot and profile\?'/)
  assert.match(pluginSource, /This will permanently delete the bot[\s\S]*?and its associated Hermes profile at[\s\S]*?This cannot be undone\./)
})
