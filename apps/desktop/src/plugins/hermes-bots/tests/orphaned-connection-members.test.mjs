import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

// #93492 root fix: deleting a connection sweeps/annotates the persisted
// group-chat member rows that referenced it (registry 'removed' push), and a
// hydrate-time pass annotates rows orphaned before the sweep existed. Rows
// are marked (sourceMissing → the existing 'Gateway removed' degraded state),
// never silently deleted.

function runtime() {
  const atom = initial => {
    let value = initial
    return { get: () => value, set: next => { value = next }, listen: () => () => undefined }
  }
  const jsx = (type, props = {}) => ({ type, props })
  const storageWrites = []
  const context = {
    atom,
    jsx,
    jsxs: jsx,
    useQuery: () => ({}),
    useValue: value => (value?.get ? value.get() : value),
    useState: value => [value, () => undefined],
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: {
      state: {
        connectionId: { get: () => 'local', listen: () => undefined },
        profile: { get: () => 'ops', listen: () => undefined }
      },
      request: () => undefined
    },
    sdk: new Proxy({}, { get: () => undefined })
  }
  const code = source
    .replace(/^import \* as sdk from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat(
      '\nglobalThis.__orphan = { $groupChats, sweepGroupChatMembersForRemovedConnection, annotateOrphanedGroupChatMembers, markOrphanedGroupMemberDescriptor, groupMemberReferencesConnection, botSourceStatus, durableGroupChatMembers, botWorkspaceOwnerKey, setBotsWorkspaceOwner, useModelOptions, pluginCtxRef: () => pluginCtx, setPluginCtx: value => { pluginCtx = value } };'
    )
  vm.runInNewContext(code, context)
  context.__storageWrites = storageWrites
  return context
}

function member(overrides = {}) {
  return {
    name: 'dixie',
    handle: 'dixie',
    connectionId: 'conn-gone',
    remoteSource: true,
    sourceScoped: true,
    route: { connectionId: 'conn-gone', mode: 'remote', profile: 'dixie', targetProfile: 'dixie' },
    ...overrides
  }
}

test('removed-connection sweep annotates matching members and persists; other rows untouched', () => {
  const { __orphan: o } = runtime()
  const writes = []
  o.setPluginCtx({ storage: { set: (key, value) => { writes.push([key, value]); return Promise.resolve() } } })

  o.$groupChats.set({
    Crew: {
      log: [{ id: '1', at: 1 }],
      watermarks: {},
      members: [member(), member({ name: 'bob', connectionId: 'conn-live', route: { connectionId: 'conn-live', mode: 'remote', profile: 'bob', targetProfile: 'bob' } })]
    }
  })

  assert.equal(o.sweepGroupChatMembersForRemovedConnection('conn-gone'), true)

  const room = o.$groupChats.get().Crew
  const swept = room.members.find(m => m.name === 'dixie')
  const kept = room.members.find(m => m.name === 'bob')

  // Annotated, not deleted: identity survives, marked degraded.
  assert.equal(room.members.length, 2)
  assert.equal(swept.sourceMissing, true)
  assert.equal(swept.sourceReachable, false)
  assert.equal(kept.sourceMissing, undefined)
  // The degraded mark renders as the existing 'Gateway removed' state.
  assert.equal(o.botSourceStatus(swept).label, 'Gateway removed')
  // Persisted so the fix survives restarts (the poisoned row lived in storage).
  assert.ok(writes.some(([key]) => key === 'group-chats'))
})

test('removed-connection sweep is idempotent and ignores blank ids', () => {
  const { __orphan: o } = runtime()
  o.setPluginCtx({ storage: { set: () => Promise.resolve() } })
  o.$groupChats.set({ Crew: { log: [], watermarks: {}, members: [member()] } })

  assert.equal(o.sweepGroupChatMembersForRemovedConnection(''), false)
  assert.equal(o.sweepGroupChatMembersForRemovedConnection('conn-gone'), true)
  // Already annotated: nothing left to change.
  assert.equal(o.sweepGroupChatMembersForRemovedConnection('conn-gone'), false)
})

test('hydrate annotate: lost-connectionId rows are marked even without a registry', () => {
  const { __orphan: o } = runtime()
  const rooms = {
    Crew: {
      log: [],
      watermarks: {},
      // The exact persisted shape from #93492: remoteSource kept, connectionId gone.
      members: [{ name: 'halakukhan', handle: 'halakukhan', connectionId: null, remoteSource: true }]
    }
  }

  const { rooms: next, changed } = o.annotateOrphanedGroupChatMembers(rooms, null)

  assert.equal(changed, true)
  assert.equal(next.Crew.members[0].sourceMissing, true)
  assert.equal(o.botSourceStatus(next.Crew.members[0]).label, 'Gateway removed')
})

test('hydrate annotate: with a live registry, members on dead connections are marked, live and local kept', () => {
  const { __orphan: o } = runtime()
  const rooms = {
    Crew: {
      log: [],
      watermarks: {},
      members: [
        member(),
        member({ name: 'bob', connectionId: 'conn-live', route: { connectionId: 'conn-live', mode: 'remote', profile: 'bob', targetProfile: 'bob' } }),
        { name: 'local-pal' }
      ]
    }
  }

  const { rooms: next, changed } = o.annotateOrphanedGroupChatMembers(rooms, new Set(['conn-live']))

  assert.equal(changed, true)
  assert.equal(next.Crew.members.find(m => m.name === 'dixie').sourceMissing, true)
  assert.equal(next.Crew.members.find(m => m.name === 'bob').sourceMissing, undefined)
  assert.equal(next.Crew.members.find(m => m.name === 'local-pal').sourceMissing, undefined)
})

test('hydrate annotate: no registry means only the unresolvable-route shape is touched', () => {
  const { __orphan: o } = runtime()
  const rooms = { Crew: { log: [], watermarks: {}, members: [member()] } }

  // conn-gone still has an id; without a registry we cannot prove it dead.
  const { changed } = o.annotateOrphanedGroupChatMembers(rooms, null)

  assert.equal(changed, false)
})

test('render-reachable route callers degrade on an orphaned row instead of throwing', () => {
  const { __orphan: o } = runtime()
  const orphaned = { name: 'halakukhan', handle: 'halakukhan', connectionId: null, remoteSource: true }

  // botWorkspaceOwnerKey (sidebar sync / Bots home / context menus)
  assert.doesNotThrow(() => o.botWorkspaceOwnerKey(orphaned))
  assert.equal(o.botWorkspaceOwnerKey(orphaned), 'bot:halakukhan')
  // setBotsWorkspaceOwner (sidebar visibility listener)
  assert.doesNotThrow(() => o.setBotsWorkspaceOwner('bot:halakukhan', orphaned))
  // durableGroupChatMembers (every group send / roster refresh)
  assert.doesNotThrow(() => o.durableGroupChatMembers([orphaned]))
  const durable = o.durableGroupChatMembers([{ ...orphaned, sourceMissing: true, sourceReachable: false }])[0]
  // The degraded mark survives the durable rebuild.
  assert.equal(durable.sourceMissing, true)
  // useModelOptions (hook body — runs during pane render)
  assert.doesNotThrow(() => o.useModelOptions(orphaned))
})

test('source contract: the sweep is wired to the registry removed push and unbound on dispose', () => {
  assert.match(source, /connections\?\.onChanged\?\.\(/)
  assert.match(source, /payload\?\.reason === 'removed'/)
  assert.match(source, /sweepGroupChatMembersForRemovedConnection\(payload\.connectionId\)/)
  assert.match(source, /unbindConnectionsChanged\(\)/)
})

test('source contract: hydrate runs the orphan annotate over persisted rooms', () => {
  assert.match(source, /annotateOrphanedGroupChatMembers\(\$groupChats\.get\(\), liveIds\)/)
})
