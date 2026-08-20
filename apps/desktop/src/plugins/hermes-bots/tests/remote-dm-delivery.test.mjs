import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// Cross-machine bot DMs: a remote @mention must land in the recipient's
// CANONICAL Bot Chat (pinned id → title → create, never a fresh session per
// mention), carry the "Message from 🤖 <sender> (@handle):" attribution
// prefix so the recipient's messaging protocol recognizes an agent-to-agent
// message, and poll for the reply so it can be relayed back.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function runtime(hostOverrides = {}) {
  const context = {
    console,
    setTimeout: fn => {
      fn()
      return 0
    },
    clearTimeout: () => undefined,
    Date,
    URL,
    atom: initial => {
      let value = initial
      return { get: () => value, set: next => (value = next), listen: () => () => undefined }
    },
    host: {
      request: async () => ({}),
      requestProfile: async () => ({}),
      notify: () => undefined,
      notifyError: () => undefined,
      state: {
        profile: { get: () => 'default', listen: () => undefined },
        connectionId: { get: () => 'local', listen: () => undefined },
        gateway: { listen: () => undefined }
      },
      ...hostOverrides
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
    .concat('\nglobalThis.__dm = { deliverRemoteRosterMentions, ensureRemoteCanonicalChat };\n')
  vm.runInNewContext(code, context, { filename: 'plugin.js' })
  return context
}

test('remote DM resumes the pinned canonical Bot Chat instead of creating a new session', async () => {
  const calls = []
  const ctx = runtime({
    requestProfile: async (route, method, params) => {
      calls.push([method, params])

      if (method === 'profiles.list') {
        return {
          profiles: [{
            name: 'dixie',
            ui_meta: { 'hermes-bots': { chat: 'stored-42' } },
            preferred_session: {
              id: 'stored-42',
              resolved_id: 'stored-42',
              title: 'Bot Chat',
              message_count: 20
            }
          }]
        }
      }

      if (method === 'session.resume' && params.session_id === 'stored-42') {
        return { session_id: 'runtime-9', session_key: 'stored-42', messages: [] }
      }

      if (method === 'session.resume') {
        return { session_id: 'runtime-9', messages: [{ role: 'assistant', content: 'done' }], inflight: false, running: false }
      }

      return {}
    }
  })

  const { runtime: rt, stored } = await ctx.__dm.ensureRemoteCanonicalChat(
    { connectionId: 'mac-mini', mode: 'remote', profile: 'dixie', targetProfile: 'dixie' },
    'dixie'
  )

  assert.equal(rt, 'runtime-9')
  assert.equal(stored, 'stored-42')
  assert.ok(!calls.some(([method]) => method === 'session.create'), 'must not mint a fresh session when the pin resumes')
})

test('remote DM fails closed on transient profile lookup without minting', async () => {
  const calls = []
  const ctx = runtime({
    requestProfile: async (_route, method) => {
      calls.push(method)
      if (method === 'profiles.list') throw new Error('gateway reconnecting')
      if (method === 'session.create') throw new Error('must not mint during an ambiguous lookup')
      return {}
    }
  })

  await assert.rejects(
    ctx.__dm.ensureRemoteCanonicalChat(
      { connectionId: 'source-a', mode: 'remote', profile: 'worker', targetProfile: 'backend-worker' },
      'worker'
    ),
    /gateway reconnecting/
  )
  assert.equal(calls.includes('session.create'), false)
})

test('remote DM fails closed on transient pinned resume without minting', async () => {
  const calls = []
  const ctx = runtime({
    requestProfile: async (_route, method) => {
      calls.push(method)
      if (method === 'profiles.list') {
        return {
          profiles: [{
            name: 'backend-worker',
            ui_meta: { 'hermes-bots': { chat: 'stored-pin' } },
            preferred_session: { id: 'stored-pin', title: 'Bot Chat', message_count: 10 }
          }]
        }
      }
      if (method === 'session.resume') throw new Error('resume unavailable')
      if (method === 'session.create') throw new Error('must not replace a confirmed pin')
      return {}
    }
  })

  await assert.rejects(
    ctx.__dm.ensureRemoteCanonicalChat(
      { connectionId: 'source-a', mode: 'remote', profile: 'worker', targetProfile: 'backend-worker' },
      'worker'
    ),
    /resume unavailable/
  )
  assert.equal(calls.includes('session.create'), false)
})

test('remote DM reuses a pinned title-drifted chat with history', async () => {
  const calls = []
  const ctx = runtime({
    requestProfile: async (_route, method, params) => {
      calls.push({ method, params })
      if (method === 'profiles.list') {
        return {
          profiles: [{
            name: 'backend-worker',
            ui_meta: { 'hermes-bots': { chat: 'stored-pin' } },
            preferred_session: {
              id: 'stored-pin',
              resolved_id: 'stored-tip',
              title: 'Investigate disk pressure',
              message_count: 44
            }
          }]
        }
      }
      if (method === 'session.resume') {
        assert.equal(params.session_id, 'stored-tip')
        return { session_id: 'runtime-tip', session_key: 'stored-pin' }
      }
      if (method === 'session.create') throw new Error('must not fork title-drifted history')
      return {}
    }
  })

  const resolved = await ctx.__dm.ensureRemoteCanonicalChat(
    { connectionId: 'source-a', mode: 'remote', profile: 'worker', targetProfile: 'backend-worker' },
    'worker'
  )
  assert.equal(resolved.runtime, 'runtime-tip')
  assert.equal(resolved.stored, 'stored-pin')
  assert.equal(calls.some(call => call.method === 'session.create'), false)
})

test('remote DM adopts the exact hidden Bot Chat before minting', async () => {
  const calls = []
  const route = { connectionId: 'source-a', mode: 'remote', profile: 'worker', targetProfile: 'backend-worker' }
  const ctx = runtime({
    requestProfile: async (capturedRoute, method, params) => {
      calls.push({ capturedRoute, method, params })
      if (method === 'profiles.list') return { profiles: [{ name: 'backend-worker', ui_meta: {} }] }
      if (method === 'session.list') {
        return { sessions: [{ id: 'existing-chat', resolved_id: 'existing-tip', title: 'Bot Chat' }] }
      }
      if (method === 'session.resume') {
        return { session_id: 'runtime-existing', session_key: 'existing-chat' }
      }
      if (method === 'session.create') throw new Error('must adopt before minting')
      return {}
    }
  })

  const resolved = await ctx.__dm.ensureRemoteCanonicalChat(route, 'worker')
  assert.equal(resolved.runtime, 'runtime-existing')
  assert.equal(resolved.stored, 'existing-chat')
  const scan = calls.find(call => call.method === 'session.list')
  assert.equal(scan.params.title, 'Bot Chat')
  assert.equal(scan.params.include_hidden, true)
  assert.equal(calls.some(call => call.method === 'session.create'), false)
})

test('remote DM persists a newly resolved canonical id through connection-qualified metadata', async () => {
  const calls = []
  const route = { connectionId: 'source-a', mode: 'remote', profile: 'worker', targetProfile: 'backend-worker' }
  const ctx = runtime({
    requestProfile: async (capturedRoute, method, params) => {
      calls.push({ capturedRoute, method, params })
      if (method === 'profiles.list') return { profiles: [{ name: 'backend-worker', ui_meta: {} }] }
      if (method === 'session.list') return { sessions: [] }
      if (method === 'session.create') return { session_id: 'runtime-new', stored_session_id: 'stored-new' }
      return {}
    }
  })

  const resolved = await ctx.__dm.ensureRemoteCanonicalChat(route, 'worker')
  assert.equal(resolved.runtime, 'runtime-new')
  assert.equal(resolved.stored, 'stored-new')
  const persisted = calls.find(call => call.method === 'profiles.configure')
  assert.equal(persisted.capturedRoute.connectionId, 'source-a')
  assert.equal(persisted.capturedRoute.profile, 'worker')
  assert.equal(persisted.params.name, 'backend-worker')
  assert.equal(persisted.params.ui_meta['hermes-bots'].chat, 'stored-new')
})

test('remote DM carries sender attribution and relays the reply', async () => {
  const submits = []
  const notices = []
  const ctx = runtime({
    requestProfile: async (route, method, params) => {
      if (method === 'profiles.list') {
        return { profiles: [{ name: 'dixie', ui_meta: {} }] }
      }

      if (method === 'session.list') {
        return { sessions: [{ id: 'stored-1', resolved_id: 'stored-1', title: 'Bot Chat' }] }
      }

      if (method === 'session.resume' && params.session_id === 'stored-1' && params.omit_messages) {
        return { session_id: 'runtime-1', session_key: 'stored-1' }
      }

      if (method === 'prompt.submit') {
        submits.push(params.text)
        return {}
      }

      if (method === 'session.resume') {
        // First (baseline) read: empty. After submit: reply present.
        return submits.length
          ? { messages: [{ role: 'user', content: 'x' }, { role: 'assistant', content: 'disk is 40% full' }], inflight: false, running: false }
          : { messages: [] }
      }

      return {}
    },
    notify: notice => notices.push(notice)
  })

  await ctx.__dm.deliverRemoteRosterMentions(
    [{ name: 'dixie', connectionId: 'mac-mini', connectionLabel: 'Mac Mini', remoteSource: true }],
    'what is the disk space?',
    { name: 'Hermes', handle: 'hermes' }
  )

  assert.equal(submits.length, 1)
  assert.match(submits[0], /^Message from 🤖 Hermes \(@hermes\): what is the disk space\?$/u)
  assert.ok(
    notices.some(notice => /disk is 40% full/.test(notice?.message || '')),
    'the recipient reply must be relayed back as a notification'
  )
})

test('remote DM preserves a non-identity targetProfile through list, resume, create, and submit', async () => {
  const calls = []
  let submitted = false
  const ctx = runtime({
    requestProfile: async (route, method, params) => {
      calls.push({ route, method, params })

      if (method === 'profiles.list') {
        return { profiles: [{ name: 'backend-worker', ui_meta: {} }] }
      }

      if (method === 'session.resume' && params.omit_messages) {
        throw new Error('not found')
      }

      if (method === 'session.create') {
        return { session_id: 'runtime-1', stored_session_id: 'stored-1' }
      }

      if (method === 'prompt.submit') {
        submitted = true
        return {}
      }

      if (method === 'session.resume') {
        return submitted
          ? { messages: [{ role: 'assistant', content: 'done' }], inflight: false, running: false }
          : { messages: [] }
      }

      return {}
    }
  })
  const route = {
    connectionId: 'remote-a',
    mode: 'remote',
    profile: 'worker',
    targetProfile: 'backend-worker'
  }

  await ctx.__dm.deliverRemoteRosterMentions(
    [{ name: 'worker', connectionId: 'remote-a', remoteSource: true, route }],
    'ping',
    { name: 'Hermes', handle: 'hermes' }
  )

  assert.equal(calls.length > 0, true)
  assert.equal(calls.every(call => call.route.profile === 'worker'), true)
  assert.equal(calls.every(call => call.route.targetProfile === 'backend-worker'), true)
  for (const call of calls.filter(call => ['session.resume', 'session.create'].includes(call.method))) {
    assert.equal(call.params.profile, 'backend-worker')
  }
  assert.equal(calls.find(call => call.method === 'prompt.submit')?.params.session_id, 'runtime-1')
})

test('remote-active DM keeps an immutable local recipient route through lookup, submit, and poll', async () => {
  const calls = []
  let submitted = false
  const ctx = runtime({
    state: {
      profile: { get: () => 'sender', listen: () => undefined },
      connectionId: { get: () => 'remote-a', listen: () => undefined },
      gateway: { listen: () => undefined }
    },
    requestProfile: async (route, method, params) => {
      calls.push({ route, method, params })

      if (method === 'profiles.list') {
        return { profiles: [{ name: 'backend-worker', ui_meta: {} }] }
      }

      if (method === 'session.resume' && params.omit_messages) {
        throw new Error('not found')
      }

      if (method === 'session.create') {
        return { session_id: 'runtime-local', stored_session_id: 'stored-local' }
      }

      if (method === 'prompt.submit') {
        submitted = true
        return {}
      }

      return submitted
        ? { messages: [{ role: 'assistant', content: 'local reply' }], inflight: false, running: false }
        : { messages: [] }
    }
  })
  const route = Object.freeze({
    connectionId: 'local',
    mode: 'local',
    profile: 'worker',
    targetProfile: 'backend-worker'
  })

  await ctx.__dm.deliverRemoteRosterMentions(
    [{ name: 'worker', connectionId: 'local', sourceScoped: true, remoteSource: true, route }],
    'ping local',
    { name: 'Remote Sender', handle: 'sender' }
  )

  assert.equal(calls.length > 0, true)
  assert.equal(calls.every(call => call.route.connectionId === 'local'), true)
  assert.equal(calls.every(call => call.route.profile === 'worker'), true)
  assert.equal(calls.every(call => call.route.targetProfile === 'backend-worker'), true)
  assert.equal(calls.some(call => call.method === 'profiles.list'), true)
  assert.equal(calls.some(call => call.method === 'prompt.submit'), true)
  assert.equal(calls.filter(call => call.method === 'session.resume').length >= 2, true)
  for (const call of calls.filter(call => ['session.resume', 'session.create'].includes(call.method))) {
    assert.equal(call.params.profile, 'backend-worker')
  }
})

test('source contract: DM poll shares the group-turn shape (bounded, new-assistant-message)', () => {
  assert.match(pluginSource, /const REMOTE_DM_TIMEOUT_MS = /)
  assert.match(pluginSource, /pollRemoteDmReply/)
  assert.match(pluginSource, /Message from \\u\{1F916\} \$\{senderName\} \(@\$\{senderHandle\}\)/)
})
