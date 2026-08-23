import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function deferred() {
  let resolve
  let reject
  const promise = new Promise((res, rej) => {
    resolve = res
    reject = rej
  })

  return { promise, resolve, reject }
}

function load({ requestProfile, agents, profileRoutes } = {}) {
  const values = new Map()
  const atom = initial => {
    const slot = {
      get: () => values.get(slot),
      set: value => values.set(slot, typeof value === 'function' ? value(values.get(slot)) : value),
      listen: () => undefined
    }
    values.set(slot, initial)
    return slot
  }
  const calls = []
  let activeConnectionId = 'remote-a'
  const host = {
    activeConnectionId: () => activeConnectionId,
    agents: agents || (async () => ({ agents: [], sources: [] })),
    deleteProfile: async route => calls.push(['deleteProfile', route]),
    ensureAgent: async (connectionId, profile) => {
      activeConnectionId = connectionId
      calls.push(['ensureAgent', connectionId, profile])
    },
    newChat: route => calls.push(['newChat', route]),
    notify: () => undefined,
    notifyError: () => undefined,
    profileRoutes: profileRoutes || (async () => []),
    openSession: async (...args) => calls.push(['openSession', ...args]),
    request: async (method, params) => {
      calls.push(['ambient', method, params])
      return {}
    },
    requestProfile: requestProfile || (async (route, method, params) => {
      calls.push(['profile', route, method, params])
      return {}
    }),
    state: {
      connectionId: { get: () => activeConnectionId, listen: () => undefined },
      gateway: { get: () => 'open', listen: () => undefined },
      profile: { get: () => 'default', listen: () => undefined }
    }
  }
  const context = {
    atom,
    clearTimeout,
    console,
    Date,
    document: { createElement: () => ({}), getElementById: () => null, head: { appendChild: () => undefined } },
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    queryClient: { invalidateQueries: () => undefined },
    setTimeout,
    host,
    window: { setTimeout, clearTimeout }
  }
  const code = source
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat(`
      globalThis.__race = {
        $botMeta,
        botConnectionRoute,
        botRosterKey,
        botSelectionKey,
        deleteBot,
        duplicateBot,
        ensureBotMetadata,
        groupMemberKey,
        migrateBotMeta,
        migratedLocalRoutes,
        newBotChat,
        openBotCanonicalChat,
        persistBotMetaSnapshot,
        prepareBotSource,
        requestForBot,
        saveBotMeta,
        setPluginStorage: storage => { pluginCtx = { storage } }
      }
    `)
  vm.runInNewContext(code, context, { filename: 'plugin.js' })
  return {
    context,
    calls,
    host,
    setActiveConnection: value => {
      activeConnectionId = value
    }
  }
}

const remoteBot = {
  connectionId: 'remote-a',
  connectionLabel: 'Remote A',
  name: 'default',
  remoteSource: true,
  route: { connectionId: 'remote-a', mode: 'remote', profile: 'default', targetProfile: 'remote-default' },
  sourceScoped: true,
  targetProfile: 'remote-default'
}

function recordingStorage(
  initial,
  failSetKey = null,
  failRemoveKeys = [],
  failSetTimes = Number.POSITIVE_INFINITY
) {
  const values = new Map(Object.entries(initial))
  const operations = []
  const failedRemovals = new Set(failRemoveKeys)
  let failedSets = 0

  return {
    operations,
    get: async key => values.get(key) ?? null,
    set: async (key, value) => {
      operations.push(['set', key, value])
      if (key === failSetKey && failedSets < failSetTimes) {
        failedSets += 1
        throw new Error('disk full')
      }
      values.set(key, value)
    },
    remove: async key => {
      operations.push(['remove', key])
      if (failedRemovals.has(key)) {
        throw new Error('remove failed')
      }
      values.delete(key)
    }
  }
}

test('same-name bots remain pinned to their own connection under concurrent requests', async () => {
  const pendingA = deferred()
  const pendingB = deferred()
  const seen = []
  const runtime = load({
    requestProfile: (route, method) => {
      seen.push([route, method])
      return route.connectionId === 'remote-a' ? pendingA.promise : pendingB.promise
    }
  })
  const botB = { ...remoteBot, connectionId: 'remote-b', route: { ...remoteBot.route, connectionId: 'remote-b', targetProfile: 'remote-default-b' } }

  const first = runtime.context.__race.requestForBot(remoteBot, 'profiles.describe')
  const second = runtime.context.__race.requestForBot(botB, 'profiles.describe')
  await runtime.host.ensureAgent('remote-b', 'default')
  pendingB.resolve({ ok: true })
  pendingA.resolve({ ok: true })
  await Promise.all([first, second])

  assert.deepEqual(seen.map(([route]) => route.connectionId), ['remote-a', 'remote-b'])
  assert.deepEqual(seen.map(([route]) => route.targetProfile), ['remote-default', 'remote-default-b'])
})

test('source-scoped routing fails closed instead of falling back to ambient host.request', async () => {
  const runtime = load()
  const ambient = []
  runtime.host.request = async (...args) => ambient.push(args)
  runtime.host.requestProfile = undefined

  await assert.rejects(
    runtime.context.__race.requestForBot(remoteBot, 'profiles.configure'),
    /Cannot route profiles\.configure/
  )
  assert.deepEqual(ambient, [])
})

test('source-scoped routing rejects a missing connection id before dispatch', async () => {
  const runtime = load()
  const incomplete = {
    ...remoteBot,
    connectionId: '',
    route: { ...remoteBot.route, connectionId: '' }
  }

  await assert.rejects(
    runtime.context.__race.requestForBot(incomplete, 'profiles.configure'),
    /no connection owner/i
  )
  assert.equal(runtime.calls.some(call => call[0] === 'ambient' || call[0] === 'profile'), false)
})

test('non-identity alias hydrates metadata from the backend row before Edit saves it', async () => {
  const calls = []
  const workerBot = {
    ...remoteBot,
    name: 'worker',
    route: { ...remoteBot.route, profile: 'worker', targetProfile: 'backend-worker' },
    targetProfile: 'backend-worker'
  }
  const runtime = load({
    requestProfile: async (_route, method, params) => {
      calls.push([method, params])
      if (method === 'profiles.list') {
        return {
          profiles: [{
            name: 'backend-worker',
            ui_meta: {
              'hermes-bots': { title: 'Hydrated', pinned: true, chat: 'worker-chat' }
            }
          }]
        }
      }
      return { applied: { ui_meta: true } }
    }
  })

  // prepareBotSource is a capability gate under name-identity — no pin returns.
  await runtime.context.__race.prepareBotSource(workerBot)

  const hydrated = await runtime.context.__race.ensureBotMetadata(workerBot)
  assert.deepEqual(JSON.parse(JSON.stringify(hydrated)), {
    title: 'Hydrated',
    pinned: true,
    chat: 'worker-chat'
  })

  await runtime.context.__race.saveBotMeta(workerBot, { title: 'Edited' })
  const edit = calls.find(([method]) => method === 'profiles.configure')
  assert.deepEqual(JSON.parse(JSON.stringify(edit[1])), {
    name: 'backend-worker',
    ui_meta: {
      'hermes-bots': { title: 'Edited', pinned: true, chat: 'worker-chat' }
    }
  })
})

test('non-identity alias resolves the canonical chat by NAME on the backend profile', async () => {
  const requests = []
  const workerBot = {
    ...remoteBot,
    name: 'worker',
    route: { ...remoteBot.route, profile: 'worker', targetProfile: 'backend-worker' },
    targetProfile: 'backend-worker'
  }
  const runtime = load({
    requestProfile: async (_route, method, params) => {
      requests.push([method, params])
      if (method === 'session.list') {
        return {
          sessions: [{ id: 'worker-chat', resolved_id: 'worker-chat-tip', title: 'Bot Chat', message_count: 3 }]
        }
      }
      if (method === 'session.create') {
        throw new Error('must not create: the registry row exists on the alias profile')
      }
      return {}
    }
  })

  const result = await runtime.context.__race.openBotCanonicalChat(workerBot)

  assert.equal(result, 'worker-chat')
  const lookup = requests.find(([method]) => method === 'session.list')
  assert.equal(lookup[1].profile, 'backend-worker', 'lookup uses the backend alias, not the logical name')
  assert.equal(lookup[1].title, 'Bot Chat')
  assert.equal(lookup[1].include_hidden, true)
  const opened = runtime.calls.find(call => call[0] === 'openSession')
  assert.equal(opened[1], 'worker-chat-tip', 'the lineage tip opens; the registry row stays the identity')
})

test('remote canonical lookup failure rejects instead of minting on the remote source', async () => {
  const runtime = load({
    requestProfile: async (_route, method) => {
      if (method === 'session.list') {
        throw new Error('remote gateway not ready')
      }
      if (method === 'session.create') {
        throw new Error('must not create: a failed remote lookup is not "no chat exists"')
      }
      return {}
    }
  })

  await assert.rejects(() => runtime.context.__race.openBotCanonicalChat(remoteBot), /Bot Chat registry/)
  assert.equal(runtime.calls.some(call => call[0] === 'openSession'), false)
})

test('delayed edit, group mutation, and canonical session RPCs retain the route', async () => {
  const pending = deferred()
  const seen = []
  const runtime = load({
    requestProfile: (route, method, params) => {
      seen.push([route, method, params])
      return pending.promise
    }
  })

  const edit = runtime.context.__race.saveBotMeta(remoteBot, { title: 'Remote title' })
  const group = runtime.context.__race.requestForBot(remoteBot, 'profiles.configure', { name: 'default' })
  const open = runtime.context.__race.openBotCanonicalChat(remoteBot)
  await runtime.host.ensureAgent('remote-b', 'default')
  pending.resolve({ applied: { ui_meta: true }, profiles: [{ name: 'default', preferred_session: { id: 'stored-remote' } }] })
  await Promise.all([edit, group, open])

  assert.ok(seen.length >= 3)
  assert.ok(seen.every(([route]) => route.connectionId === 'remote-a'))
  assert.ok(seen.every(([route]) => Object.isFrozen(route)))
})

test('delayed duplicate, delete, and new chat keep source ownership', async () => {
  const pending = deferred()
  const runtime = load({
    requestProfile: (route, method) => {
      runtime.calls.push(['profile', route, method])
      return pending.promise
    }
  })
  const workerBot = {
    ...remoteBot,
    name: 'worker',
    route: { ...remoteBot.route, profile: 'worker', targetProfile: 'backend-worker' },
    targetProfile: 'backend-worker'
  }
  runtime.context.__race.$botMeta.set({ 'remote-a::worker': { title: 'Original' } })

  const duplicate = runtime.context.__race.duplicateBot(workerBot, [workerBot])
  const edit = runtime.context.__race.saveBotMeta(workerBot, { pinned: true })
  await runtime.host.ensureAgent('remote-b', 'default')
  pending.resolve({ profiles: [{ name: 'default' }], applied: { ui_meta: true } })
  await Promise.all([duplicate, edit])

  runtime.context.__race.newBotChat(workerBot)
  const deletePromise = runtime.context.__race.deleteBot(workerBot)
  await deletePromise

  const routes = runtime.calls
    .filter(call => call[0] === 'profile' || call[0] === 'deleteProfile' || call[0] === 'newChat')
    .map(call => call[1])
    .filter(value => value && typeof value === 'object')
  assert.ok(routes.length >= 3)
  assert.ok(routes.every(route => route.connectionId === 'remote-a'))
})


test('default deletion is rejected in the executable mutation layer', async () => {
  const runtime = load()
  await assert.rejects(runtime.context.__race.deleteBot(remoteBot), /default profile cannot be deleted/i)
  assert.equal(runtime.calls.some(call => call[0] === 'deleteProfile'), false)
})

test('source-scoped bot metadata save commits v2 marker after its data', async () => {
  const storage = recordingStorage({ 'bot-meta-v2-migrated': true })
  const runtime = load({
    requestProfile: async () => ({ applied: { ui_meta: true } })
  })
  runtime.context.__race.setPluginStorage(storage)

  await runtime.context.__race.saveBotMeta(remoteBot, { title: 'Remote' })

  assert.deepEqual(storage.operations.map(operation => operation.slice(0, 2)), [
    ['remove', 'bot-meta-v2-migrated'],
    ['set', 'bot-meta-v2'],
    ['set', 'bot-meta-v2-migrated']
  ])
})

test('scoped metadata reconciliation commits v2 marker after its data', async () => {
  const storage = recordingStorage({ 'bot-meta-v2-migrated': true })
  const runtime = load()
  runtime.context.__race.setPluginStorage(storage)

  await runtime.context.__race.persistBotMetaSnapshot({
    'remote-a::default': { hidden: true }
  }, true)

  assert.deepEqual(storage.operations.map(operation => operation.slice(0, 2)), [
    ['remove', 'bot-meta-v2-migrated'],
    ['set', 'bot-meta-v2'],
    ['set', 'bot-meta-v2-migrated']
  ])
})

test('v1 metadata migrates only for a provable sole-local topology and keeps rollback data', async () => {
  const writes = []
  const runtime = load({
    agents: async () => ({
      agents: [{ connectionId: 'local', connectionKind: 'local', profile: 'default' }],
      sources: [{ connectionId: 'local', kind: 'local' }]
    }),
    profileRoutes: async () => [{ connectionId: 'local', mode: 'local', profile: 'default', targetProfile: 'default' }],
    storage: {
      get: async key => key === 'bot-meta' ? { default: { title: 'Local' } } : null,
      set: async (key, value) => writes.push([key, value])
    }
  })

  const migrated = await runtime.context.__race.migrateBotMeta({
    get: async key => key === 'bot-meta' ? { default: { title: 'Local' } } : null,
    remove: async () => undefined,
    set: async (key, value) => writes.push([key, value])
  })
  assert.equal(migrated, true)
  assert.deepEqual(Object.keys(runtime.context.__race.$botMeta.get()), ['local::default'])
  assert.deepEqual(writes.map(([key]) => key), ['bot-meta-v2', 'bot-meta-v2-migrated'])
})

test('migration rolls back routes when a later v1 profile has no local route', async () => {
  const runtime = load({
    agents: async () => ({
      agents: [{ connectionId: 'local', connectionKind: 'local', profile: 'first' }],
      sources: [{ connectionId: 'local', kind: 'local' }]
    }),
    profileRoutes: async () => [
      { connectionId: 'local', mode: 'local', profile: 'first', targetProfile: 'backend-first' }
    ]
  })


  const migrated = await runtime.context.__race.migrateBotMeta({
    get: async key => key === 'bot-meta'
      ? { first: { title: 'First' }, missing: { title: 'Missing' } }
      : null,
    set: async () => assert.fail('storage must not be written before every route is proven')
  })

  assert.equal(migrated, false)
  assert.equal(runtime.context.__race.migratedLocalRoutes.size, 0)
  assert.deepEqual(JSON.parse(JSON.stringify(runtime.context.__race.$botMeta.get())), {
    first: { title: 'First' },
    missing: { title: 'Missing' }
  })
})

test('failed normal v2 successor commit preserves the last committed generation on reload', async () => {
  const committed = {
    'remote-a::default': { title: 'B' }
  }
  const storage = recordingStorage(
    {
      'bot-meta-v2': committed,
      'bot-meta-v2-migrated': true
    },
    'bot-meta-v2-migrated',
    [],
    1
  )
  const runtime = load({
    requestProfile: async () => ({ applied: { ui_meta: true } }),
    storage
  })
  runtime.context.__race.setPluginStorage(storage)
  assert.equal(await runtime.context.__race.migrateBotMeta(storage), true)

  await runtime.context.__race.saveBotMeta(remoteBot, { title: 'D' })

  const reloaded = load({ storage })
  assert.equal(await reloaded.context.__race.migrateBotMeta(storage), true)
  assert.deepEqual(
    JSON.parse(JSON.stringify(reloaded.context.__race.$botMeta.get())),
    committed
  )
})

test('failed marker commit rolls persisted v2 back and reload keeps v1 authoritative', async () => {
  const storage = recordingStorage(
    { 'bot-meta': { first: { title: 'First' } } },
    'bot-meta-v2-migrated'
  )
  const options = {
    agents: async () => ({
      agents: [{ connectionId: 'local', connectionKind: 'local', profile: 'first' }],
      sources: [{ connectionId: 'local', kind: 'local' }]
    }),
    profileRoutes: async () => [
      { connectionId: 'local', mode: 'local', profile: 'first', targetProfile: 'backend-first' }
    ],
    storage
  }
  const runtime = load(options)

  const migrated = await runtime.context.__race.migrateBotMeta(storage)

  assert.equal(migrated, false)
  assert.equal(runtime.context.__race.migratedLocalRoutes.size, 0)
  assert.deepEqual(JSON.parse(JSON.stringify(runtime.context.__race.$botMeta.get())), {
    first: { title: 'First' }
  })
  assert.deepEqual(storage.operations.map(operation => operation.slice(0, 2)), [
    ['remove', 'bot-meta-v2-migrated'],
    ['set', 'bot-meta-v2'],
    ['set', 'bot-meta-v2-migrated'],
    ['remove', 'bot-meta-v2-migrated'],
    ['remove', 'bot-meta-v2']
  ])

  const reloaded = load(options)
  const acceptedAfterReload = await reloaded.context.__race.migrateBotMeta(storage)

  assert.equal(acceptedAfterReload, false)
  assert.deepEqual(JSON.parse(JSON.stringify(reloaded.context.__race.$botMeta.get())), {
    first: { title: 'First' }
  })
})

test('failed marker and failed v2 cleanup still reload v1 instead of markerless v2', async () => {
  const storage = recordingStorage(
    { 'bot-meta': { first: { title: 'Rollback' } } },
    'bot-meta-v2-migrated',
    ['bot-meta-v2']
  )
  const options = {
    agents: async () => ({
      agents: [{ connectionId: 'local', connectionKind: 'local', profile: 'first' }],
      sources: [{ connectionId: 'local', kind: 'local' }]
    }),
    profileRoutes: async () => [
      { connectionId: 'local', mode: 'local', profile: 'first', targetProfile: 'backend-first' }
    ],
    storage
  }
  const runtime = load(options)

  assert.equal(await runtime.context.__race.migrateBotMeta(storage), false)

  const reloaded = load(options)
  assert.equal(await reloaded.context.__race.migrateBotMeta(storage), false)
  assert.deepEqual(JSON.parse(JSON.stringify(reloaded.context.__race.$botMeta.get())), {
    first: { title: 'Rollback' }
  })
})

test('crash-window v2 write without a marker reloads v1 rollback data', async () => {
  const storage = recordingStorage({
    'bot-meta': { first: { title: 'Rollback' } },
    'bot-meta-v2': { 'local::first': { title: 'Crash window' } }
  })
  const runtime = load({ storage })

  assert.equal(await runtime.context.__race.migrateBotMeta(storage), false)
  assert.deepEqual(JSON.parse(JSON.stringify(runtime.context.__race.$botMeta.get())), {
    first: { title: 'Rollback' }
  })
})

test('v1 metadata is not projected onto same-name bots in a multi-source topology', async () => {
  const writes = []
  const runtime = load({
    agents: async () => ({
      agents: [
        { connectionId: 'local', connectionKind: 'local', profile: 'default' },
        { connectionId: 'remote-a', connectionKind: 'remote', profile: 'default' }
      ],
      sources: [
        { connectionId: 'local', kind: 'local' },
        { connectionId: 'remote-a', kind: 'remote' }
      ]
    }),
    profileRoutes: async () => [
      { connectionId: 'local', mode: 'local', profile: 'default', targetProfile: 'default' },
      remoteBot.route
    ]
  })

  const migrated = await runtime.context.__race.migrateBotMeta({
    get: async key => key === 'bot-meta' ? { default: { title: 'Legacy local' } } : null,
    set: async (key, value) => writes.push([key, value])
  })

  assert.equal(migrated, false)
  assert.deepEqual(JSON.parse(JSON.stringify(runtime.context.__race.$botMeta.get())), {
    default: { title: 'Legacy local' }
  })
  assert.deepEqual(writes, [])
})

test('bot-meta-v2 is authoritative only with its persisted commit marker', async () => {
  const runtime = load()
  const v2 = {
    'remote-a::default': { title: 'A' },
    'remote-b::default': { title: 'B' }
  }

  const migrated = await runtime.context.__race.migrateBotMeta({
    get: async key => key === 'bot-meta-v2'
      ? v2
      : key === 'bot-meta-v2-migrated'
        ? true
        : null,
    set: async () => assert.fail('existing v2 must not be rewritten during hydration')
  })

  assert.equal(migrated, true)
  assert.deepEqual(JSON.parse(JSON.stringify(runtime.context.__race.$botMeta.get())), v2)
})

test('markerless bot-meta-v2 is ignored on reload', async () => {
  const runtime = load()
  const markerlessV2 = { 'remote-a::default': { title: 'Uncommitted' } }

  const migrated = await runtime.context.__race.migrateBotMeta({
    get: async key => key === 'bot-meta-v2' ? markerlessV2 : null,
    set: async () => assert.fail('markerless v2 must not be committed implicitly')
  })

  assert.equal(migrated, false)
  assert.deepEqual(JSON.parse(JSON.stringify(runtime.context.__race.$botMeta.get())), {})
})

test('selection and metadata keys do not collide for same-name connections', () => {
  const runtime = load()
  const { botRosterKey, botSelectionKey, botConnectionRoute } = runtime.context.__race
  const botB = {
    ...remoteBot,
    connectionId: 'remote-b',
    route: { ...remoteBot.route, connectionId: 'remote-b' }
  }
  const a = botConnectionRoute(remoteBot)
  const b = botConnectionRoute(botB)

  assert.notEqual(botRosterKey({ ...remoteBot, route: a }), botRosterKey({ ...botB, route: b }))
  assert.notEqual(botSelectionKey({ ...remoteBot, route: a }), botSelectionKey({ ...botB, route: b }))
})
