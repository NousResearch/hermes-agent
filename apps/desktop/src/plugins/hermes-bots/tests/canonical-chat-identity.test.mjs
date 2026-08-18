import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

// ── canonical open-path harness (slice: createCanonicalChat + openBotCanonicalChat)
function loadOpenPath({ openSession, request }) {
  const start = source.indexOf('const canonicalCreations = new Map()')
  const end = source.indexOf('function displayName(', start)
  const saved = []
  const requests = []
  const context = {
    host: {
      openSession,
      request: async (method, params) => {
        requests.push({ method, params })
        return request(method, params)
      }
    },
    saveBotMeta: (name, patch) => saved.push({ name, patch: JSON.parse(JSON.stringify(patch)) }),
    $hideBotChats: { get: () => false },
    window: { setTimeout: callback => callback() }
  }
  const section = source
    .slice(start, end)
    .concat('\nglobalThis.__open = { openBotCanonicalChat };\n')

  assert.notEqual(start, -1, 'canonical creation section is missing')
  assert.notEqual(end, -1, 'canonical creation section delimiter is missing')
  vm.runInNewContext(section, context, { filename: 'canonical-open.js' })
  return { ...context.__open, saved, requests, host: context.host }
}

const HISTORY = { id: 'hist-1', title: 'Weekly review', preview: 'history preview', last_active: 1000 }

// ── grandfather: no pin + existing history adopts the previewed session ────

test('grandfather: no pin + history opens and pins THAT session, no new chat', async () => {
  const runtime = loadOpenPath({
    openSession: async () => undefined,
    request: async () => ({})
  })

  const result = await runtime.openBotCanonicalChat('ops', null, HISTORY)

  assert.equal(result, 'hist-1')
  assert.deepEqual(runtime.saved, [{ name: 'ops', patch: { chat: 'hist-1' } }])
  assert.equal(runtime.requests.some(r => r.method === 'session.create'), false,
    'must not mint a new chat when the previewed session can be adopted')
})

test('grandfather: no pin + no history keeps the creation flow', async () => {
  const runtime = loadOpenPath({
    openSession: async () => undefined,
    request: async method =>
      method === 'session.create' ? { stored_session_id: 'stored-1', session_id: 'runtime-1' } : {}
  })

  const result = await runtime.openBotCanonicalChat('ops', null, null)

  assert.equal(result, 'stored-1')
  assert.equal(runtime.requests.some(r => r.method === 'session.create'), true)
})

test('grandfather: adoption hydration failure surfaces without forking a replacement chat', async () => {
  const runtime = loadOpenPath({
    openSession: async id => {
      if (id === 'hist-1') throw new Error('session vanished')
    },
    request: async method =>
      method === 'session.create' ? { stored_session_id: 'stored-2', session_id: 'runtime-2' } : {}
  })

  await assert.rejects(runtime.openBotCanonicalChat('ops', null, HISTORY), /session vanished/)
  assert.equal(runtime.saved.some(s => s.patch?.chat === 'hist-1'), false,
    'a failed adoption must not persist the dead id as the pin')
  assert.equal(runtime.requests.some(r => r.method === 'session.create'), false,
    'a transient hydration failure must not fork the canonical chat')
})

// ── precise pin verification (no session.list pagination/hidden semantics) ─

test('pin: preferred_session present opens the resolved session and keeps the pin', async () => {
  const opened = []
  const runtime = loadOpenPath({
    openSession: async (id, options) => { opened.push({ id, options }) },
    request: async method => {
      if (method === 'profiles.list') {
        return {
          profiles: [{
            name: 'ops',
            preferred_session: {
              id: 'pin-1', resolved_id: 'pin-1', title: 'Bot Chat',
              preview: 'latest', started_at: 1, last_active: 2, message_count: 3
            }
          }]
        }
      }
      return {}
    }
  })

  const result = await runtime.openBotCanonicalChat('ops', 'pin-1', HISTORY)

  assert.equal(result, 'pin-1')
  assert.deepEqual(JSON.parse(JSON.stringify(opened)), [{
    id: 'pin-1',
    options: {
      profile: 'ops',
      intent: 'main',
      awaitHydration: true,
      expectHistory: true
    }
  }])
  assert.equal(runtime.saved.length, 0, 'a live pin must not be rewritten')
  assert.equal(runtime.requests.some(r => r.method === 'session.create'), false)
  // The pin is verified through the precise resolver, never session.list.
  assert.equal(runtime.requests.some(r => r.method === 'session.list'), false)
})

test('pin: compression-rotated pin opens the live tip, keeps the durable pin', async () => {
  const opened = []
  const runtime = loadOpenPath({
    openSession: async id => { opened.push(id) },
    request: async method => {
      if (method === 'profiles.list') {
        return {
          profiles: [{
            name: 'ops',
            preferred_session: {
              id: 'root-1', resolved_id: 'tip-9', title: 'Bot Chat',
              preview: 'post-compression', started_at: 1, last_active: 9, message_count: 42
            }
          }]
        }
      }
      return {}
    }
  })

  const result = await runtime.openBotCanonicalChat('ops', 'root-1', HISTORY)

  assert.deepEqual(opened, ['tip-9'])
  assert.equal(result, 'root-1', 'the stored pin keeps its durable identity')
  assert.equal(runtime.saved.length, 0)
})

test('pin: definitively gone pin re-pins to the previewed session, not rows[0]', async () => {
  const runtime = loadOpenPath({
    openSession: async () => undefined,
    request: async method => {
      if (method === 'profiles.list') {
        return { profiles: [{ name: 'ops', preferred_session: null }] }
      }
      return {}
    }
  })

  const result = await runtime.openBotCanonicalChat('ops', 'dead-pin', HISTORY)

  assert.equal(result, 'hist-1')
  assert.deepEqual(runtime.saved, [{ name: 'ops', patch: { chat: 'hist-1' } }])
  assert.equal(runtime.requests.some(r => r.method === 'session.create'), false)
})

test('pin: gone pin + no history clears the pin and creates', async () => {
  const runtime = loadOpenPath({
    openSession: async () => undefined,
    request: async method => {
      if (method === 'profiles.list') {
        return { profiles: [{ name: 'ops', preferred_session: null }] }
      }
      if (method === 'session.create') return { stored_session_id: 'stored-3', session_id: 'runtime-3' }
      return {}
    }
  })

  const result = await runtime.openBotCanonicalChat('ops', 'dead-pin', null)

  assert.equal(result, 'stored-3')
  // Pin cleared first (dead pin is provably unusable), then the freshly
  // created chat pins itself inside createCanonicalChat.
  assert.deepEqual(runtime.saved, [
    { name: 'ops', patch: { chat: null } },
    { name: 'ops', patch: { chat: 'stored-3' } }
  ])
})

test('pin: precise hit but failed hydration keeps the pin and surfaces the failure', async () => {
  const runtime = loadOpenPath({
    openSession: async () => { throw new Error('socket hiccup') },
    request: async method => {
      if (method === 'profiles.list') {
        return {
          profiles: [{
            name: 'ops',
            preferred_session: {
              id: 'pin-1', resolved_id: 'pin-1', title: 'Bot Chat',
              preview: 'latest', started_at: 1, last_active: 2, message_count: 3
            }
          }]
        }
      }
      return {}
    }
  })

  await assert.rejects(runtime.openBotCanonicalChat('ops', 'pin-1', HISTORY), /socket hiccup/)
  assert.equal(runtime.saved.length, 0, 'a confirmed-live pin must survive a transient open failure')
  assert.equal(runtime.requests.some(r => r.method === 'session.create'), false,
    'must not fork the forever-chat on a hiccup')
})

// ── transient failures must never destroy the pin ──────────────────────────

test('transient: profiles.list failure keeps the pin when the direct open works', async () => {
  const runtime = loadOpenPath({
    openSession: async () => undefined,
    request: async method => {
      if (method === 'profiles.list') throw new Error('gateway reconnecting')
      return {}
    }
  })

  const result = await runtime.openBotCanonicalChat('ops', 'pin-1', HISTORY)

  assert.equal(result, 'pin-1')
  assert.equal(runtime.saved.length, 0, 'a hiccup must not clear or rewrite the pin')
  assert.equal(runtime.requests.some(r => r.method === 'session.create'), false,
    'a hiccup must not mint a replacement chat')
})

test('transient: profiles.list failure + failed direct open preserves pin and surfaces Retry', async () => {
  const runtime = loadOpenPath({
    openSession: async id => {
      if (id === 'pin-1') throw new Error('resume rejected')
    },
    request: async method => {
      if (method === 'profiles.list') throw new Error('gateway reconnecting')
      if (method === 'session.create') return { stored_session_id: 'stored-4', session_id: 'runtime-4' }
      return {}
    }
  })

  await assert.rejects(runtime.openBotCanonicalChat('ops', 'pin-1', HISTORY), /resume rejected/)
  assert.deepEqual(runtime.saved, [], 'an inconclusive outage must never clear the canonical pin')
  assert.equal(runtime.requests.some(r => r.method === 'session.create'), false,
    'an inconclusive outage must never fork the canonical chat')
})

// ── preferred_session_ids request shaping (pure helper) ────────────────────

function loadHelpers() {
  const atom = value => ({ get: () => value, set: () => undefined })
  const jsx = (type, props = {}) => ({ type, props })
  const context = {
    atom,
    jsx,
    jsxs: jsx,
    useQuery: () => ({}),
    useValue: value => (value?.get ? value.get() : value),
    useState: value => [value, () => undefined],
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: { state: { profile: { get: () => 'ops', listen: () => undefined } }, request: () => undefined }
  }
  const code = source
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat('\nglobalThis.__preferredSessionIds = preferredSessionIds;')
  vm.runInNewContext(code, context)
  return context
}

test('preferredSessionIds: collects only live pins', () => {
  // vm-realm objects fail assert.deepEqual prototype checks — compare via JSON.
  const collect = meta => JSON.parse(JSON.stringify(loadHelpers().__preferredSessionIds(meta)))
  assert.deepEqual(
    collect({ ops: { chat: 'pin-1' }, scribe: { chat: null }, chef: { title: 'Chef' } }),
    { ops: 'pin-1' }
  )
  assert.deepEqual(collect({}), {})
  assert.deepEqual(collect(undefined), {})
})
