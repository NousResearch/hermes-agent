import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// Cross-connection Bot Mode: the New Agent "Create on" picker targets another
// registered connection's backend, and group chats seat members from other
// machines by routing their turns through host.requestProfile route
// descriptors instead of the active gateway.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function runtime() {
  const context = {
    console,
    setTimeout,
    clearTimeout,
    Date,
    URL,
    atom: initial => {
      const listeners = new Set()
      let value = initial
      return {
        get: () => value,
        set: next => {
          value = next
          listeners.forEach(fn => fn(next))
        },
        listen: fn => {
          listeners.add(fn)
          return () => listeners.delete(fn)
        }
      }
    },
    host: {
      request: async () => ({}),
      requestProfile: async () => ({}),
      state: {
        profile: { get: () => 'default', listen: () => undefined },
        connectionId: { get: () => 'local', listen: () => undefined },
        gateway: { listen: () => undefined }
      }
    },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } }
  }
  const code = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat(
      '\nglobalThis.__x = { botConnectionRoute, requestForBot, groupMemberKey, parseGroupChatMentions, resolveGroupResponders, formatGroupChatLine, buildGroupChatTurnPrompt, canCreateGroupChat, loadMultiSourceRoster };\n'
    )
  vm.runInNewContext(code, context, { filename: 'plugin.js' })
  return context
}

test('botConnectionRoute: remote rows get a route descriptor, local rows do not', () => {
  const ctx = runtime()
  const { botConnectionRoute } = ctx.__x

  assert.equal(botConnectionRoute({ name: 'writer', connectionId: 'local' }), null)
  assert.equal(botConnectionRoute({ name: 'writer' }), null)
  // remoteSource is required — an annotated ACTIVE row keeps the active door.
  assert.equal(botConnectionRoute({ name: 'writer', connectionId: 'spark', sourceScoped: true }), null)

  const route = botConnectionRoute({ name: 'dixie', connectionId: 'mac-mini', remoteSource: true })
  assert.deepEqual(JSON.parse(JSON.stringify(route)), {
    connectionId: 'mac-mini',
    mode: 'remote',
    profile: 'dixie',
    targetProfile: 'dixie'
  })

  const thisDevice = botConnectionRoute({
    name: 'lokay',
    connectionId: 'local',
    connectionKind: 'local',
    remoteSource: true
  })
  assert.deepEqual(JSON.parse(JSON.stringify(thisDevice)), {
    connectionId: 'local',
    mode: 'local',
    profile: 'lokay',
    targetProfile: 'lokay'
  })
})

test('requestForBot: remote members go through requestProfile, local through host.request', async () => {
  const ctx = runtime()
  const calls = []
  ctx.host.request = async (method, params) => {
    calls.push(['active', method, params])
    return {}
  }
  ctx.host.requestProfile = async (route, method, params) => {
    calls.push(['routed', route.connectionId, route.mode, method, params])
    return {}
  }

  const { requestForBot } = ctx.__x

  await requestForBot({ name: 'local-bot' }, 'session.create', { title: 'x' })
  await requestForBot({ name: 'dixie', connectionId: 'mac-mini', remoteSource: true }, 'prompt.submit', { text: 'hi' })
  await requestForBot(
    { name: 'lokay', connectionId: 'local', connectionKind: 'local', remoteSource: true },
    'prompt.submit',
    { text: 'hello from the active remote gateway' }
  )

  assert.equal(calls[0][0], 'active')
  assert.equal(calls[0][1], 'session.create')
  assert.equal(calls[1][0], 'routed')
  assert.equal(calls[1][1], 'mac-mini')
  assert.equal(calls[1][2], 'remote')
  assert.equal(calls[1][3], 'prompt.submit')
  assert.equal(calls[2][0], 'routed')
  assert.equal(calls[2][1], 'local')
  assert.equal(calls[2][2], 'local')
  assert.equal(calls[2][3], 'prompt.submit')
})

test('group creation is enabled for one active bot plus one bot on another connection', () => {
  const ctx = runtime()

  assert.equal(
    ctx.__x.canCreateGroupChat([
      { name: 'marvin', connectionId: 'house-of-marvin' },
      { name: 'default', connectionId: 'local', remoteSource: true }
    ]),
    true
  )
})

test('group creation stays disabled when the union roster has only one bot', () => {
  const ctx = runtime()

  assert.equal(ctx.__x.canCreateGroupChat([{ name: 'marvin', connectionId: 'house-of-marvin' }]), false)
})

test('groupMemberKey: local members keep bare names (persisted-room compat), remote members are source-qualified', () => {
  const ctx = runtime()
  const { groupMemberKey } = ctx.__x

  assert.equal(groupMemberKey({ name: 'ops' }), 'ops')
  assert.equal(groupMemberKey({ name: 'dixie', connectionId: 'mac-mini', remoteSource: true }), 'mac-mini::dixie')
})

test('mentions: @name-device handle resolves the remote member; same-named agents stay distinct', () => {
  const ctx = runtime()
  const { parseGroupChatMentions, resolveGroupResponders } = ctx.__x
  const members = [
    { name: 'dixie', title: '' },
    { name: 'dixie', handle: 'dixie-mac-mini', connectionId: 'mac-mini', remoteSource: true }
  ]

  const parsed = parseGroupChatMentions('hey @dixie-mac-mini can you check disk space', members)
  assert.deepEqual([...parsed.mentioned], ['mac-mini::dixie'])

  const log = [{ from: { kind: 'user', name: 'You' }, text: '@dixie-mac-mini ping', at: 1 }]
  const responders = resolveGroupResponders(log, members)
  assert.equal(responders.length, 1)
  assert.equal(responders[0].remoteSource, true)
})

test('room lines and turn prompts badge cross-connection speakers with their device', () => {
  const ctx = runtime()
  const { formatGroupChatLine, buildGroupChatTurnPrompt } = ctx.__x

  const line = formatGroupChatLine(
    { from: { kind: 'member', name: 'dixie', source: 'Mac Mini' }, text: 'done', at: 1 },
    'ops'
  )
  assert.equal(line, 'dixie [Mac Mini]: done')

  const prompt = buildGroupChatTurnPrompt({
    groupName: 'infra',
    members: [
      { name: 'ops', title: '' },
      { name: 'dixie', connectionId: 'mac-mini', connectionLabel: 'Mac Mini', remoteSource: true }
    ],
    viewer: { name: 'ops' },
    deltaLines: ['You (user): hi']
  })
  assert.match(prompt, /@dixie \[on Mac Mini\]/)
})

test('source contract: New Agent has a Create on picker that routes creation to the target backend', () => {
  assert.match(pluginSource, /'Create on'/)
  assert.match(pluginSource, /const \[targetConnection, setTargetConnection\] = useState\(''\)/)
  // Creation and configuration go through the target door, not bare host.request.
  assert.match(pluginSource, /await requestForTarget\('profiles\.create', \{/)
  assert.match(pluginSource, /await requestForTarget\('profiles\.configure', \{ name: slug/)
  // The picker only renders on multi-connection registries.
  assert.match(pluginSource, /connections\.length > 1/)
})

test('source contract: group chat turns route through requestForBot on the member source', () => {
  assert.match(pluginSource, /await requestForBot\(member, 'prompt\.submit'/)
  assert.match(pluginSource, /await requestForBot\(member, 'session\.create'/)
  // Room records persist remote member descriptors.
  assert.match(pluginSource, /members: Array\.isArray\(room\.members\) \? room\.members : \[\]/)
})


test('cold Sessions roster merges This device while a remote gateway is active', async () => {
  const ctx = runtime()
  ctx.host.request = async (method, params) => {
    assert.equal(method, 'profiles.list')
    assert.deepEqual(JSON.parse(JSON.stringify(params)), { include_sessions: false })
    return { profiles: [{ name: 'default' }] }
  }
  ctx.host.agents = async () => ({
    primaryConnectionId: 'house',
    agents: [
      { profile: 'default', connectionId: 'house', connectionKind: 'remote', connectionLabel: 'HouseOfMarvin' },
      { profile: 'lokay', connectionId: 'local', connectionKind: 'local', connectionLabel: 'This device', handle: 'lokay-this-device' }
    ]
  })

  const loaded = await ctx.__x.loadMultiSourceRoster('house', { include_sessions: false })
  const profiles = JSON.parse(JSON.stringify(loaded.profiles))

  assert.equal(profiles.length, 2)
  assert.equal(profiles[0].name, 'default')
  assert.equal(profiles[0].connectionId, 'house')
  assert.equal(profiles[1].name, 'lokay')
  assert.equal(profiles[1].connectionId, 'local')
  assert.equal(profiles[1].remoteSource, true)
  assert.equal(profiles[1].handle, 'lokay-this-device')
})

test('regression: host.connections() result is normalized for BOTH SDK shapes before the picker gate', () => {
  // Current SDKs return the registry ROWS from host.connections() (the
  // documented contract); desktops that predate the SDK-side unwrap resolve
  // the raw IPC payload — the registry OBJECT ({version, primary,
  // connections: [...]}). The picker gate (Array.isArray(connections) &&
  // length > 1) needs rows either way, so the plugin must accept both
  // shapes. Pinning one shape only reopens #89823 on the other.
  assert.match(
    pluginSource,
    /setConnections\(Array\.isArray\(value\) \? value : Array\.isArray\(value\?\.connections\) \? value\.connections : \[\]\)/
  )
})
