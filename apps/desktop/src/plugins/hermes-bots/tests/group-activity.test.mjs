import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// Collapsible group Activity view: a runtime-only, bounded feed of truthful
// turn events (queued / working / replied / passed / timed-out / failed /
// cancelled / settled / delivered) that the room shows in a quiet disclosure.
// The transcript stays the only durable record — activity is never persisted.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

/** Load the plugin in a vm with a scripted gateway so member turns are
 *  deterministic. `turnScript(profile, prompt, n, session)` returns the
 *  member's reply text, a promise of it, or throws for a failed turn. */
function load(turnScript = () => '(pass)') {
  const values = new Map()
  const atom = initial => {
    const slot = { get: () => values.get(slot), set: value => values.set(slot, value) }
    values.set(slot, initial)
    return slot
  }
  const calls = []
  const sessions = new Map()
  const runtimeToStored = new Map()
  const titleToStored = new Map()
  let sessionSequence = 0

  const resolveSession = (profile, target) => {
    const stored = runtimeToStored.get(target) || (sessions.has(target) ? target : titleToStored.get(`${profile}::${target}`))
    return stored ? sessions.get(stored) : null
  }
  const context = {
    atom,
    setTimeout: fn => {
      fn()
      return 0
    },
    clearTimeout: () => undefined,
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: {
      request: async (method, params) => {
        if (method === 'session.create') {
          sessionSequence += 1
          const stored = `sid-${params.profile}-${sessionSequence}`
          const runtime = `rt-${params.profile}-${sessionSequence}`
          const session = { stored, runtime, profile: params.profile, title: params.title, messages: [] }
          sessions.set(stored, session)
          runtimeToStored.set(runtime, stored)
          titleToStored.set(`${params.profile}::${params.title}`, stored)
          return { session_id: runtime, stored_session_id: stored, message_count: 0, messages: [] }
        }
        if (method === 'session.resume') {
          const session = resolveSession(params.profile, params.session_id)
          if (!session) {
            // Shaped like the real gateway's JsonRpcGatewayError (`.code`,
            // apps/shared/src/json-rpc-gateway.ts) so ensureGroupChatSession's
            // fail-closed `.code !== 4007` check sees the same "genuinely
            // not found" signal production does.
            const err = new Error(`session not found: ${params.session_id}`)
            err.code = 4007
            throw err
          }
          return {
            session_id: session.runtime,
            session_key: session.stored,
            message_count: session.messages.length,
            messages: [...session.messages],
            inflight: false,
            running: false
          }
        }
        if (method === 'prompt.submit') {
          const session = resolveSession(null, params.session_id)
          if (!session) {
            throw new Error(`runtime session not found: ${params.session_id}`)
          }
          session.messages.push({ role: 'user', content: params.text })
          calls.push({
            profile: session.profile,
            prompt: params.text,
            runtime: session.runtime,
            stored: session.stored,
            title: session.title
          })
          const reply = await turnScript(session.profile, params.text, calls.length, session)
          session.messages.push({ role: 'assistant', content: reply })
          return {}
        }
        return {}
      },
      state: { profile: { get: () => 'default', listen: () => undefined }, gateway: { listen: () => undefined } },
      notify: () => undefined,
      notifyError: () => undefined
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
      '\nglobalThis.__ga = { sendToGroupChat, recordGroupActivity, currentGroupActivity, groupActivityLabel, updateGroupChat, $groupChats, $groupActivity, $botAttention, GROUP_ACTIVITY_LIMIT };\n'
    )
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  const storageWrites = new Map()
  context.plugin.register({
    storage: { get: () => null, set: (key, value) => storageWrites.set(key, value) },
    register: () => undefined
  })
  return { ...context.__ga, calls, sessions, storageWrites }
}

const MEMBERS = [{ name: 'research', title: '' }, { name: 'builder', title: '' }, { name: 'ops', title: 'The Ops' }]

async function drain(gc, group) {
  for (let i = 0; i < 200 && (gc.$groupChats.get()[group] || {}).running; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }
}

function feed(gc, group) {
  return gc.$groupActivity.get()[group]?.events || []
}

test('a settled turn records the full truthful arc: queued, working, replies and passes, settled', async () => {
  const gc = load((profile, prompt) => {
    if (profile === 'research' && !prompt.includes('(you)')) {
      return 'I looked into it — ship it.'
    }
    return '(pass)'
  })

  gc.sendToGroupChat('Room', MEMBERS, 'please review')
  await drain(gc, 'Room')

  const events = feed(gc, 'Room')
  const kinds = events.map(e => e.kind)
  assert.equal(kinds[0], 'queued', 'the send lands first')
  assert.equal(kinds[kinds.length - 1], 'settled', 'the run ends with a settle')
  assert.equal(kinds.filter(k => k === 'working').length >= 3, true, 'every member on turn shows working')
  assert.equal(kinds.includes('replied'), true, 'a real reply records replied')
  assert.equal(kinds.filter(k => k === 'passed').length >= 2, true, 'silent members record passed')
  const replied = events.find(e => e.kind === 'replied')
  assert.equal(replied.member, 'research')
})

test('a failed member turn records failed instead of a phantom reply', async () => {
  const gc = load(profile => {
    if (profile === 'builder') {
      throw new Error('gateway hiccup')
    }
    return '(pass)'
  })

  gc.sendToGroupChat('Flaky', MEMBERS, 'anyone around?')
  await drain(gc, 'Flaky')

  const events = feed(gc, 'Flaky')
  assert.equal(events.some(e => e.kind === 'failed' && e.member === 'builder'), true)
})

test('a failed member turn preserves and prefers its typed reason', async () => {
  const gc = load(profile => {
    if (profile === 'builder') {
      const error = new Error('generic gateway failure')
      error.data = { reason: 'provider_auth_or_access' }
      throw error
    }
    return '(pass)'
  })

  gc.sendToGroupChat('Typed failure', MEMBERS, 'anyone around?')
  await drain(gc, 'Typed failure')

  const failed = feed(gc, 'Typed failure').find(e => e.kind === 'failed' && e.member === 'builder')
  assert.equal(failed.reason, 'provider_auth_or_access')
  assert.equal(Object.values(gc.$botAttention.get())[0]?.reason, 'provider_auth_or_access')
})

test('an untyped failed member turn keeps the message-classification fallback', async () => {
  const gc = load(profile => {
    if (profile === 'builder') {
      throw new Error('No LLM provider configured')
    }
    return '(pass)'
  })

  gc.sendToGroupChat('Untyped failure', MEMBERS, 'anyone around?')
  await drain(gc, 'Untyped failure')

  const failed = feed(gc, 'Untyped failure').find(e => e.kind === 'failed' && e.member === 'builder')
  assert.equal(failed.reason, undefined)
  assert.equal(Object.values(gc.$botAttention.get())[0]?.reason, 'missing_config')
})

test('a newer send interrupts the previous run and records cancelled in the CURRENT epoch', async () => {
  const gates = [null, null]
  const gate = n => {
    gates[n] = gates[n] || { promise: null, resolve: null }
    if (!gates[n].promise) {
      gates[n].promise = new Promise(resolve => {
        gates[n].resolve = resolve
      })
    }
    return gates[n].promise
  }
  const gc = load((profile, prompt, n) => gate(n))
  const member = [{ name: 'research', title: '' }]

  gc.sendToGroupChat('Busy', member, 'first ask')
  for (let i = 0; i < 50 && gc.calls.length < 1; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }
  gc.sendToGroupChat('Busy', member, 'second ask, supersede')
  for (let i = 0; i < 50 && gc.calls.length < 2; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }

  gates[2].resolve('from the new run')
  await drain(gc, 'Busy')
  gates[1].resolve('late from the old run')
  await new Promise(resolve => setImmediate(resolve))
  await new Promise(resolve => setImmediate(resolve))

  const epoch = (gc.$groupChats.get().Busy || {}).epoch || 0
  const events = feed(gc, 'Busy')
  assert.equal(events.some(e => e.kind === 'cancelled'), true, 'the superseded loop reports its own interruption')
  // The view shows only the current run: every visible event is this epoch.
  assert.equal(gc.currentGroupActivity('Busy').every(e => (e.epoch || 0) === epoch), true)
  assert.equal(gc.currentGroupActivity('Busy').some(e => e.kind === 'cancelled'), true)
})

test('epoch filtering drops events from a superseded run', () => {
  const gc = load()
  gc.updateGroupChat('Epoch', r => {
    r.log = []
    r.epoch = 5
    return r
  })
  gc.recordGroupActivity('Epoch', { kind: 'queued', member: 'You' })
  gc.recordGroupActivity('Epoch', { kind: 'working', member: 'research' })

  gc.updateGroupChat('Epoch', r => {
    r.epoch = 6
    return r
  })

  assert.equal(gc.currentGroupActivity('Epoch').length, 0, 'old-run events drop out of view')

  gc.recordGroupActivity('Epoch', { kind: 'working', member: 'research' })
  const current = gc.currentGroupActivity('Epoch')
  assert.equal(current.length, 1)
  assert.equal(current[0].kind, 'working')
})

test('the feed is bounded and never grows past GROUP_ACTIVITY_LIMIT', () => {
  const gc = load()
  gc.updateGroupChat('Cap', r => {
    r.log = []
    return r
  })

  for (let i = 0; i < gc.GROUP_ACTIVITY_LIMIT + 25; i++) {
    gc.recordGroupActivity('Cap', { kind: 'working', member: 'research' })
  }

  assert.equal(feed(gc, 'Cap').length, gc.GROUP_ACTIVITY_LIMIT)
})

test('activity is runtime-only: never persisted, never hydrated', async () => {
  const gc = load()
  gc.sendToGroupChat('Volatile', MEMBERS, 'hello')
  await drain(gc, 'Volatile')

  assert.equal(gc.storageWrites.has('group-activity'), false, 'no activity storage key is ever written')
  assert.doesNotMatch(pluginSource, /['"]group-activity['"]/, 'the plugin never reads or writes an activity key')
})

test('labels read like a person wrote them, with settled/cancelled as room-level lines', () => {
  const gc = load()
  assert.equal(gc.groupActivityLabel({ kind: 'queued', member: 'You' }), 'You sent a message')
  assert.equal(gc.groupActivityLabel({ kind: 'replied', member: 'research' }), 'research replied')
  assert.equal(gc.groupActivityLabel({ kind: 'timed-out', member: 'ops' }), 'ops took too long')
  assert.equal(gc.groupActivityLabel({ kind: 'cancelled', member: null }), 'turn interrupted by a newer message')
  assert.equal(gc.groupActivityLabel({ kind: 'settled', member: null }), 'turn settled')
})

test('source contract: the workspace mounts a quiet, collapsed-by-default disclosure', () => {
  // Collapsed default — opening is always an explicit user action.
  assert.match(pluginSource, /const \[activityOpen, setActivityOpen\] = useState\(false\)/)
  // Accessible disclosure pattern.
  assert.match(pluginSource, /'aria-expanded': activityOpen/)
  assert.match(pluginSource, /'aria-controls': `group-activity:\$\{group\}`/)
  // The panel body never steals focus and never auto-scrolls.
  const panel = pluginSource.slice(pluginSource.indexOf('// Activity disclosure:'), pluginSource.indexOf('const submit = () => {'))
  assert.doesNotMatch(panel, /autoFocus/)
  assert.doesNotMatch(panel, /autoScroll|scrollIntoView/)
  // Truthful event vocabulary is wired.
  assert.match(pluginSource, /'timed-out': 'took too long'/)
  assert.match(pluginSource, /cancelled: 'turn interrupted by a newer message'/)
  // The panel sits inside the workspace between the header and the log.
  const workspace = pluginSource.slice(pluginSource.indexOf('function GroupChatWorkspace('), pluginSource.indexOf('function GroupChatMainView('))
  assert.match(workspace, /header,\s*\n\s*activityPanel,/s)
})
