import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')
const plain = value => JSON.parse(JSON.stringify(value))

function load({ profile = 'ops', request, openSession } = {}) {
  const values = new Map()
  const atom = initial => {
    const slot = {
      get: () => values.get(slot),
      set: value => values.set(slot, typeof value === 'function' ? value(values.get(slot)) : value)
    }
    values.set(slot, initial)
    return slot
  }
  const calls = []
  const host = {
    state: {
      profile: { get: () => profile, listen: () => undefined },
      gateway: { get: () => 'open', listen: () => undefined }
    },
    request: async (method, params) => {
      calls.push([method, params])
      return request?.(method, params)
    },
    openSession: async (...args) => {
      calls.push(['openSession', ...args])
      return openSession?.(...args)
    },
    newChat: (...args) => calls.push(['newChat', ...args]),
    notify: () => undefined,
    notifyError: () => undefined
  }
  const context = {
    atom,
    host,
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } }
  }
  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat(`\nglobalThis.__sessions = {
      openBotSessionsWorkspace,
      openProfileSession,
      filterProfileSessions,
      $botSessionsWorkspace,
      $botSelectedSessions,
      $sessionsGatewayGeneration,
      handleSessionsGatewayTransition
      };\n`)
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  return { ...context, calls }
}

test('sessions workspace: selecting the secondary workspace does not navigate', () => {
  const runtime = load()
  runtime.__sessions.openBotSessionsWorkspace({ name: 'ops' })
  assert.equal(runtime.__sessions.$botSessionsWorkspace.get(), 'ops')
  assert.equal(runtime.calls.some(([method]) => method === 'openSession'), false)
})

test('sessions workspace: invalid profile names are ignored', () => {
  const runtime = load()
  runtime.__sessions.openBotSessionsWorkspace({ name: '../ops' })
  assert.equal(runtime.__sessions.$botSessionsWorkspace.get(), null)
})

test('sessions workspace: filtering searches title, preview, and source without privileged rows', () => {
  const { filterProfileSessions } = load().__sessions
  const rows = [
    { id: 'named', title: 'Oversight', preview: 'ordinary user session', source: 'user' },
    { id: 'deploy', title: 'Deploy API', preview: 'shipping', source: 'cli' },
    { id: 'docs', title: 'Write docs', preview: 'guide', source: 'desktop' }
  ]
  assert.deepEqual(plain(filterProfileSessions(rows, '').map(row => row.id)), ['named', 'deploy', 'docs'])
  assert.deepEqual(plain(filterProfileSessions(rows, 'ship').map(row => row.id)), ['deploy'])
  assert.deepEqual(plain(filterProfileSessions(rows, 'DESKTOP').map(row => row.id)), ['docs'])
})

test('sessions workspace: opening a stored row uses profile-aware navigation and records selection', async () => {
  const runtime = load({ profile: 'default' })
  await runtime.__sessions.openProfileSession('ops', 'stored-123', 0)
  assert.deepEqual(plain(runtime.calls), [['openSession', 'stored-123', { profile: 'ops' }]])
  assert.equal(runtime.__sessions.$botSelectedSessions.get().ops, 'stored-123')
})

test('sessions workspace: malformed profile or session input is a no-op', async () => {
  const runtime = load()
  await runtime.__sessions.openProfileSession('../ops', 'stored-123', 0)
  await runtime.__sessions.openProfileSession('ops', '', 0)
  assert.deepEqual(runtime.calls, [])
})

test('sessions workspace: a gateway lifecycle change clears selection and rejects stale row clicks', async () => {
  const runtime = load()
  runtime.__sessions.$botSelectedSessions.set({ ops: 'stored-123' })
  runtime.__sessions.handleSessionsGatewayTransition()
  assert.equal(runtime.__sessions.$sessionsGatewayGeneration.get(), 1)
  assert.deepEqual(plain(runtime.__sessions.$botSelectedSessions.get()), {})

  await runtime.__sessions.openProfileSession('ops', 'stored-123', 0)
  assert.deepEqual(runtime.calls, [])
})

test('sessions workspace: an in-flight open cannot restore selection after gateway replacement', async () => {
  let runtime
  runtime = load({
    openSession: async () => runtime.__sessions.handleSessionsGatewayTransition()
  })

  await runtime.__sessions.openProfileSession('ops', 'stored-123', 0)
  assert.equal(runtime.calls.length, 1)
  assert.deepEqual(plain(runtime.__sessions.$botSelectedSessions.get()), {})
})

test('source contract: bot rows and Active now activate the owner before canonical chat', () => {
  assert.match(pluginSource, /async function prepareBotSource\(bot, pinnedChat\)/)
  assert.match(pluginSource, /await host\.ensureAgent\(bot\.connectionId, bot\.name\)/)
  assert.match(pluginSource, /host\.request\('profiles\.list', \{\}\)/)
  assert.match(pluginSource, /const open = async \(\) => \{[\s\S]*await prepareBotSource\(bot, pinnedChat\)[\s\S]*openBotCanonicalChat\(bot\.name, pinnedChat, bot\.last_session\)/)
  assert.match(pluginSource, /onOpen: bot => \{[\s\S]*await prepareBotSource\(bot, pinnedChat\)[\s\S]*openBotCanonicalChat\(bot\.name, pinnedChat, bot\.last_session\)/)
  assert.match(pluginSource, /openBotSessionsWorkspace\(bot\)[\s\S]*children: 'Sessions'/)
})

test('source contract: workspaces disclose the bounded recent-session inventory', () => {
  assert.match(pluginSource, /queryKey: \[ID, 'profile-sessions', botName, gatewayGeneration\]/)
  assert.match(pluginSource, /enabled: Boolean\(botName\)/)
  assert.match(pluginSource, /host\.request\('session\.list', \{ profile: botName, limit: PROFILE_SESSION_LIST_LIMIT, include_hidden: true \}\)/)
  assert.match(pluginSource, /Showing the \$\{PROFILE_SESSION_LIST_LIMIT\} most recent sessions\./)
  assert.match(pluginSource, /No matching sessions in the \$\{PROFILE_SESSION_LIST_LIMIT\} most recent\./)
  assert.doesNotMatch(pluginSource, /children: 'Activate profile'/)
  assert.match(pluginSource, /host\.state\.gateway\.listen\(handleSessionsGatewayTransition\)/)
})

test('source contract: session controls expose filter and selection state to assistive technology', () => {
  assert.match(pluginSource, /'aria-label': 'Filter sessions'/)
  assert.match(pluginSource, /'aria-current': active \? 'page' : undefined/)
})
