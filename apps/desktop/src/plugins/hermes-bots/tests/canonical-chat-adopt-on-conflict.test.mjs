import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// ADOPT-BEFORE-MINT (#92473 part 2): between the registry miss and our eager
// session.title write, another writer can take the canonical title (peer dm
// minting server-side, a second machine, cross-connection sync). UNIQUE(title)
// rejects our write with "already in use". Before this fix that rejection was
// read as "old gateway" and the compat path prompted into OUR stray lazy
// session — forking the forever chat. Now the mint re-consults the registry
// and adopts the winner; the stray zero-message session is abandoned to the
// gateway's pruner.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadCanonicalCreation({ openSession, request }) {
  const start = source.indexOf('const canonicalCreations = new Map()')
  const end = source.indexOf('function displayName(', start)
  const context = {
    host: { openSession, request },
    backendTargetProfile: (route, name) => route?.targetProfile || name,
    botOwner: name => (typeof name === 'string'
      ? { bot: { name }, key: name, name, route: null }
      : { bot: name, key: name?.name, name: name?.name, route: name?.route || null }),
    requestForBot: (_bot, method, params) => context.host.request(method, params),
    botWorkspaceOwnerKey: bot => String(bot?.name || bot || ''),
    window: { setTimeout: callback => callback() }
  }
  const section = source
    .slice(start, end)
    .concat('\nglobalThis.__canonical = { createCanonicalChat };\n')

  assert.notEqual(start, -1, 'canonical creation section is missing')
  assert.notEqual(end, -1, 'canonical creation section delimiter is missing')
  vm.runInNewContext(section, context, { filename: 'canonical-adopt.js' })
  return { ...context.__canonical }
}

test('title-uniqueness rejection adopts the racing winner instead of forking', async () => {
  const events = []
  let lists = 0
  const runtime = loadCanonicalCreation({
    openSession: async id => events.push(`open:${id}`),
    request: async method => {
      events.push(method)
      if (method === 'session.list') {
        lists += 1
        // First consult: registry miss (this is why we mint at all). Second
        // consult (after the conflict): the racing winner exists.
        if (lists === 1) return { sessions: [] }
        return { sessions: [{ id: 'winner-1', resolved_id: 'winner-1', title: 'Bot Chat', message_count: 3 }] }
      }
      if (method === 'session.create') return { stored_session_id: 'stray-1', session_id: 'rt-stray' }
      if (method === 'session.title') {
        throw new Error("Title 'Bot Chat' is already in use by session winner-1")
      }
      return {}
    }
  })

  assert.equal(await runtime.createCanonicalChat('ops'), 'winner-1')
  // The winner is opened; the stray is never prompted into or opened.
  assert.ok(events.includes('open:winner-1'), `winner opened (events: ${events})`)
  assert.ok(!events.includes('prompt.submit'), 'no prompt into the stray session')
  assert.ok(!events.includes('open:stray-1'), 'stray session never mounted')
})

test('a NON-conflict title failure still takes the compat path (old gateways)', async () => {
  const events = []
  const runtime = loadCanonicalCreation({
    openSession: async id => events.push(`open:${id}`),
    request: async (method) => {
      events.push(method)
      if (method === 'session.list') return { sessions: [] }
      if (method === 'session.create') return { stored_session_id: 'stored-1', session_id: 'rt-1' }
      if (method === 'session.title') throw new Error('unknown method')
      return {}
    }
  })

  assert.equal(await runtime.createCanonicalChat('ops'), 'stored-1')
  // Old gateway: eager title unsupported → the kickoff persists the lazy row.
  assert.ok(events.includes('prompt.submit'), 'compat kickoff persists the session')
  // Only the initial registry consult — a plain failure must not re-list.
  assert.equal(events.filter(e => e === 'session.list').length, 1)
})
