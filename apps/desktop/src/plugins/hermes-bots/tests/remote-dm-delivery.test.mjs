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
        return { profiles: [{ name: 'dixie', ui_meta: { 'hermes-bots': { chat: 'stored-42' } } }] }
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

test('remote DM carries sender attribution and relays the reply', async () => {
  const submits = []
  const notices = []
  const ctx = runtime({
    requestProfile: async (route, method, params) => {
      if (method === 'profiles.list') {
        return { profiles: [] }
      }

      if (method === 'session.resume' && params.session_id === 'Bot Chat' && params.omit_messages) {
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

test('source contract: DM poll shares the group-turn shape (bounded, new-assistant-message)', () => {
  assert.match(pluginSource, /const REMOTE_DM_TIMEOUT_MS = /)
  assert.match(pluginSource, /pollRemoteDmReply/)
  assert.match(pluginSource, /Message from \\u\{1F916\} \$\{senderName\} \(@\$\{senderHandle\}\)/)
})
