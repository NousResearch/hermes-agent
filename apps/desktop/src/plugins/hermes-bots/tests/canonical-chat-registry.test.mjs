import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// ── The canonical-chat REGISTRY contract ────────────────────────────────────
//
// A bot's forever-chat has exactly ONE identity: the session titled "Bot Chat"
// on that bot's profile. The core UNIQUE(title) index makes (profile,
// "Bot Chat") an exact registry — at most one row, resolved fresh on every
// open via `session.list { title: 'Bot Chat', include_hidden: true }`.
//
// There is NO session-id pin. The previous design stored a pointer in
// ui_meta['hermes-bots'].chat and spent five hardening waves (#88690, #90732,
// #90751, #91791-revert, #92042) guarding its failure modes: rows[0] steals,
// last_session adoptions, transient clears, drifted-title welds. Every "lost
// canonical chat" incident traced to that pointer dangling and a later guard
// then welding the wrong session in. Name-as-identity removes the failure
// class instead of guarding it: a name cannot dangle.
//
// This suite pins the whole contract:
//   1. open = registry lookup → open the row (lineage tip)
//   2. no row → create (adopt-before-mint lives inside creation)
//   3. no pointer is ever read or written on the open path

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadOpenPath({ openSession, request }) {
  const start = source.indexOf('const canonicalCreations = new Map()')
  const end = source.indexOf('function displayName(', start)
  const requests = []
  const opened = []
  const context = {
    host: {
      openSession: async (id, options) => {
        opened.push({ id, options })
        return openSession ? openSession(id, options) : undefined
      },
      request: async (method, params) => {
        requests.push({ method, params: JSON.parse(JSON.stringify(params ?? null)) })
        return request(method, params)
      }
    },
    // Owner-shape helpers (plugin scope): local bots carry no route, so the
    // harness resolves everything onto the ambient host.request — the exact
    // legacy single-connection behavior these tests pin.
    botOwner: owner => (typeof owner === 'string'
      ? { bot: { name: owner }, key: owner, name: owner, route: null }
      : { bot: owner, key: owner?.name, name: owner?.name, route: owner?.route || null }),
    backendTargetProfile: (route, name) => route?.targetProfile || name,
    requestForBot: (_bot, method, params) => context.host.request(method, params),
    window: { setTimeout: callback => callback() }
  }
  const section = source
    .slice(start, end)
    .concat('\nglobalThis.__open = { createCanonicalChat, openBotCanonicalChat, findExistingCanonicalChat };\n')

  assert.notEqual(start, -1, 'canonical section is missing')
  assert.notEqual(end, -1, 'canonical section delimiter is missing')
  vm.runInNewContext(section, context, { filename: 'canonical-registry.js' })
  return { ...context.__open, requests, opened }
}

// ── 1. the registry row wins, always ────────────────────────────────────────

test('open resolves the profile\u2019s "Bot Chat" row by exact title and opens it', async () => {
  const runtime = loadOpenPath({
    request: async method => {
      if (method === 'session.list') {
        return { sessions: [{ id: 'forever-chat', title: 'Bot Chat', message_count: 930 }] }
      }
      if (method === 'session.create') {
        throw new Error('must not create: the registry row exists')
      }
      return {}
    }
  })

  assert.equal(await runtime.openBotCanonicalChat('ops'), 'forever-chat')
  assert.equal(runtime.opened.length, 1)
  assert.equal(runtime.opened[0].id, 'forever-chat')
  assert.equal(runtime.opened[0].options.profile, 'ops')
  assert.equal(runtime.opened[0].options.keepAllProfilesScope, false,
    'opening a bot moves the workspace onto that bot')

  const list = runtime.requests.find(r => r.method === 'session.list')
  assert.equal(list?.params?.title, 'Bot Chat', 'lookup is by exact title')
  assert.equal(list?.params?.profile, 'ops')
  assert.equal(list?.params?.include_hidden, true,
    'canonical chats are always hidden — the lookup must see hidden rows')
})

test('a compression-rotated registry row opens the lineage tip', async () => {
  const runtime = loadOpenPath({
    request: async method => {
      if (method === 'session.list') {
        return {
          sessions: [{ id: 'root-1', resolved_id: 'tip-9', root_title: 'Bot Chat', title: 'Bot Chat', message_count: 400 }]
        }
      }
      return {}
    }
  })

  assert.equal(await runtime.openBotCanonicalChat('ops'), 'root-1',
    'the durable registry id is returned')
  assert.equal(runtime.opened[0].id, 'tip-9', 'the live tip is what opens')
})

test('the open path never reads or writes a stored pointer', () => {
  const start = source.indexOf('const canonicalCreations = new Map()')
  const end = source.indexOf('function displayName(', start)
  const section = source.slice(start, end)

  assert.doesNotMatch(section, /saveBotMeta/, 'no pointer writes on the canonical path')
  assert.doesNotMatch(section, /meta\??\.chat\b/, 'no pointer reads on the canonical path')
  assert.doesNotMatch(section, /preferred_session_ids/, 'no id-verification RPC on the canonical path')
})

test('openBotCanonicalChat takes only the bot owner — identity needs nothing else', () => {
  // The owner is the bot's name (local) or its roster row carrying the
  // immutable connection route (remote). Still no pins, no session ids.
  assert.match(source, /async function openBotCanonicalChat\(owner\) \{/)
})

// ── 2. no registry row → create ─────────────────────────────────────────────

test('no registry row mints a hidden "Bot Chat" session with the intro kickoff', async () => {
  const runtime = loadOpenPath({
    request: async method => {
      if (method === 'session.list') return { sessions: [] }
      if (method === 'session.create') return { stored_session_id: 'fresh-1', session_id: 'rt-1' }
      return {}
    }
  })

  assert.equal(await runtime.openBotCanonicalChat('newbie'), 'fresh-1')
  const create = runtime.requests.find(r => r.method === 'session.create')
  assert.equal(create?.params?.title, 'Bot Chat')
  assert.equal(create?.params?.hidden, true)
  const kickoff = runtime.requests.find(r => r.method === 'prompt.submit')
  assert.equal(kickoff?.params?.session_id, 'rt-1')
})

test('a failed open of the registry row surfaces instead of forking a replacement', async () => {
  const runtime = loadOpenPath({
    openSession: async () => {
      throw new Error('backend restarting')
    },
    request: async method => {
      if (method === 'session.list') {
        return { sessions: [{ id: 'forever-chat', title: 'Bot Chat', message_count: 12 }] }
      }
      if (method === 'session.create') {
        throw new Error('must not create: a transient open failure is not ownership loss')
      }
      return {}
    }
  })

  await assert.rejects(() => runtime.openBotCanonicalChat('ops'), /backend restarting/)
})

// ── 3. ordinary sessions are never claimed ──────────────────────────────────

test('an ordinary titled session never satisfies the registry lookup', async () => {
  const runtime = loadOpenPath({
    request: async method => {
      if (method === 'session.list') {
        // A misbehaving/older gateway ignores the title param and returns a
        // windowed listing — the local exact-title scan still applies.
        return {
          sessions: [
            { id: 'scratch', title: 'help me with x', message_count: 40 },
            { id: 'draft', title: '', message_count: 0 }
          ]
        }
      }
      if (method === 'session.create') return { stored_session_id: 'fresh-2', session_id: 'rt-2' }
      return {}
    }
  })

  assert.equal(await runtime.openBotCanonicalChat('ops'), 'fresh-2',
    'no row titled "Bot Chat" → create; never adopt an ordinary conversation')
  assert.ok(!runtime.opened.some(o => o.id === 'scratch'))
})

// ── 4. a failed lookup fails CLOSED — never "no chat exists" ────────────────
//
// The post-update window: the desktop restarts every profile backend, the
// first bot click races the warm-up, and the registry lookup RPC fails
// transiently. Swallowing that error and returning null made the failure
// indistinguishable from "this bot has no Bot Chat yet", so the create path
// minted a fresh forever-chat while the real one (data intact, hidden) still
// held the canonical title — read by users as "my bot lost everything after
// the update". A lookup failure must surface (the open paths toast
// "try again"), never resolve to mint.

test('a failed registry lookup rejects instead of minting a replacement chat', async () => {
  const runtime = loadOpenPath({
    request: async method => {
      if (method === 'session.list') {
        throw new Error('gateway not ready')
      }
      if (method === 'session.create') {
        throw new Error('must not create: a failed lookup is not "no chat exists"')
      }
      return {}
    }
  })

  await assert.rejects(() => runtime.openBotCanonicalChat('ops'), /Bot Chat registry/)
  assert.ok(!runtime.requests.some(r => r.method === 'session.create'),
    'session.create must never fire off a failed lookup')
  assert.equal(runtime.opened.length, 0)
})

test('createCanonicalChat also refuses to mint when the adoption lookup fails', async () => {
  const runtime = loadOpenPath({
    request: async method => {
      if (method === 'session.list') {
        throw new Error('backend warming up')
      }
      if (method === 'session.create') {
        throw new Error('must not create: adoption check failed, ownership is unknown')
      }
      return {}
    }
  })

  await assert.rejects(() => runtime.createCanonicalChat('ops'), /Bot Chat registry/)
  assert.ok(!runtime.requests.some(r => r.method === 'session.create'))
})
