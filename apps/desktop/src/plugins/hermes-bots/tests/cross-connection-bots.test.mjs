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
      '\nglobalThis.__x = { botConnectionRoute, requestForBot, groupMemberKey, parseGroupChatMentions, resolveGroupResponders, formatGroupChatLine, buildGroupChatTurnPrompt };\n'
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
})

test('requestForBot: remote members go through requestProfile, local through host.request', async () => {
  const ctx = runtime()
  const calls = []
  ctx.host.request = async (method, params) => {
    calls.push(['active', method, params])
    return {}
  }
  ctx.host.requestProfile = async (route, method, params) => {
    calls.push(['routed', route.connectionId, method, params])
    return {}
  }

  const { requestForBot } = ctx.__x

  await requestForBot({ name: 'local-bot' }, 'session.create', { title: 'x' })
  await requestForBot({ name: 'dixie', connectionId: 'mac-mini', remoteSource: true }, 'prompt.submit', { text: 'hi' })

  assert.equal(calls[0][0], 'active')
  assert.equal(calls[0][1], 'session.create')
  assert.equal(calls[1][0], 'routed')
  assert.equal(calls[1][1], 'mac-mini')
  assert.equal(calls[1][2], 'prompt.submit')
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

test('regression: host.connections() registry object is unwrapped before the picker gate', () => {
  // host.connections() resolves the IPC handler hermes:connections:list, which
  // returns the registry OBJECT ({version, primary, connections: [...]}) — not
  // a bare array. The picker gate (Array.isArray(connections) && length > 1)
  // never fired, so the "Create on" picker stayed hidden on multi-connection
  // desktops. The plugin must unwrap .connections before storing.
  assert.match(
    pluginSource,
    /setConnections\(Array\.isArray\(value\?\.connections\) \? value\.connections : \[\]\)/
  )
  // The built-in Connections UI consumes the registry object, so the IPC
  // handler contract must NOT change to a bare array.
  assert.doesNotMatch(pluginSource, /setConnections\(Array\.isArray\(value\) \? value : \[\]\)/)
})
