import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// Regression suite for the infinite-fork loop (hermes-agent#88200 follow-up):
// the core UNIQUE title index means at most one session per profile db holds
// the "Bot Chat" title. When a fork squats it, every later mint's title is
// silently dropped, the LLM titler renames the untitled row, and the next
// title-based identity check misreads the fresh chat as "not plumbing" —
// clearing the pin and minting again, forever. Two invariants kill the loop:
//  1. createCanonicalChat ADOPTS an existing "Bot Chat" row before creating.
//  2. A pin that resolves to a NON-plumbing session with real history is the
//     user's conversation — keep it; only an empty stray draft is replaced.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadOpenPath({ openSession, request }) {
  const start = source.indexOf('const canonicalCreations = new Map()')
  const end = source.indexOf('function displayName(', start)
  const saved = []
  const requests = []
  const context = {
    host: {
      openSession,
      request: async (method, params) => {
        requests.push({ method, params: JSON.parse(JSON.stringify(params ?? null)) })
        return request(method, params)
      }
    },
    saveBotMeta: (name, patch) => saved.push({ name, patch: JSON.parse(JSON.stringify(patch)) }),
    $hideBotChats: { get: () => false },
    window: { setTimeout: callback => callback() }
  }
  const section = source
    .slice(start, end)
    .concat('\nglobalThis.__open = { createCanonicalChat, openBotCanonicalChat };\n')

  assert.notEqual(start, -1, 'canonical creation section is missing')
  assert.notEqual(end, -1, 'canonical creation section delimiter is missing')
  vm.runInNewContext(section, context, { filename: 'canonical-adopt.js' })
  return { ...context.__open, saved, requests }
}

// ── invariant 1: adopt-before-mint ──────────────────────────────────────────

test('createCanonicalChat adopts an existing hidden "Bot Chat" row instead of creating', async () => {
  const runtime = loadOpenPath({
    openSession: async () => undefined,
    request: async method => {
      if (method === 'session.list') {
        return {
          sessions: [
            { id: 'newer-ordinary', title: 'help me with x', message_count: 12 },
            { id: 'real-forever-chat', title: 'Bot Chat', message_count: 930 }
          ]
        }
      }
      if (method === 'session.create') {
        throw new Error('must not create: the profile already owns a Bot Chat')
      }
      return {}
    }
  })

  assert.equal(await runtime.createCanonicalChat('ops'), 'real-forever-chat')
  assert.deepEqual(runtime.saved, [{ name: 'ops', patch: { chat: 'real-forever-chat' } }])
  const list = runtime.requests.find(r => r.method === 'session.list')
  assert.equal(list?.params?.include_hidden, true,
    'adoption scan must see hidden rows — canonical chats are always hidden')
})

test('createCanonicalChat still creates when no Bot Chat row exists', async () => {
  const runtime = loadOpenPath({
    openSession: async () => undefined,
    request: async method => {
      if (method === 'session.list') {
        return { sessions: [{ id: 'ordinary', title: 'help me with x', message_count: 3 }] }
      }
      if (method === 'session.create') return { stored_session_id: 'fresh-1', session_id: 'rt-1' }
      return {}
    }
  })

  assert.equal(await runtime.createCanonicalChat('newbie'), 'fresh-1')
})

test('createCanonicalChat mints when the adoption scan fails (older gateway)', async () => {
  const runtime = loadOpenPath({
    openSession: async () => undefined,
    request: async method => {
      if (method === 'session.list') throw new Error('unknown method')
      if (method === 'session.create') return { stored_session_id: 'fresh-2', session_id: 'rt-2' }
      return {}
    }
  })

  assert.equal(await runtime.createCanonicalChat('legacy'), 'fresh-2')
})

// ── invariant 2: a resolving pin with history is never abandoned ────────────

test('pin resolving to a renamed session WITH history keeps the pin (no fork)', async () => {
  const opened = []
  const runtime = loadOpenPath({
    openSession: async id => opened.push(id),
    request: async method => {
      if (method === 'profiles.list') {
        return {
          profiles: [{
            name: 'ops',
            preferred_session: {
              id: 'grandfathered', resolved_id: 'grandfathered',
              root_title: 'Use computer use to inspect…', title: 'Use computer use to inspect…',
              message_count: 930
            }
          }]
        }
      }
      if (method === 'session.create') throw new Error('must not fork a chat with 930 messages')
      return {}
    }
  })

  assert.equal(await runtime.openBotCanonicalChat('ops', 'grandfathered', null), 'grandfathered')
  assert.equal(opened.includes('grandfathered'), true)
  assert.deepEqual(runtime.saved, [], 'pin must not be cleared or rewritten')
})

test('pin resolving to an EMPTY stray draft is replaced via adoption, not a blind mint', async () => {
  const runtime = loadOpenPath({
    openSession: async () => undefined,
    request: async method => {
      if (method === 'profiles.list') {
        return {
          profiles: [{
            name: 'ops',
            preferred_session: { id: 'stray', resolved_id: 'stray', root_title: 'Untitled', title: 'Untitled', message_count: 0 }
          }]
        }
      }
      if (method === 'session.list') {
        return { sessions: [{ id: 'real-forever-chat', title: 'Bot Chat', message_count: 42 }] }
      }
      if (method === 'session.create') throw new Error('must adopt the existing Bot Chat')
      return {}
    }
  })

  assert.equal(await runtime.openBotCanonicalChat('ops', 'stray', null), 'real-forever-chat')
  assert.deepEqual(runtime.saved, [
    { name: 'ops', patch: { chat: null } },
    { name: 'ops', patch: { chat: 'real-forever-chat' } }
  ])
})
