import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

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
    botWorkspaceOwnerKey: bot => `bot:local::${bot.name}`,
    requestForBot: (_bot, method, params) => context.host.request(method, params),
    window: { setTimeout: callback => callback() }
  }
  const section = source
    .slice(start, end)
    .concat('\nglobalThis.__canonical = { createCanonicalChat };\n')

  assert.notEqual(start, -1, 'canonical creation section is missing')
  assert.notEqual(end, -1, 'canonical creation section delimiter is missing')
  vm.runInNewContext(section, context, { filename: 'canonical-creation.js' })
  return { ...context.__canonical }
}

test('regression: creation materializes and titles the lazy row before opening it', async () => {
  const events = []
  const runtime = loadCanonicalCreation({
    openSession: async id => events.push(`open:${id}`),
    request: async (method, params) => {
      events.push(method)
      if (method === 'session.create') return { stored_session_id: 'stored-1', session_id: 'runtime-1' }
      if (method === 'session.title') {
        // vm-realm objects fail assert.deepEqual on prototype identity —
        // compare via JSON.stringify (harness contract). A throw here is
        // NOT inert: createCanonicalChat reads it as "eager title
        // unsupported" and falls back to the compat kickoff.
        assert.equal(JSON.stringify(params), JSON.stringify({ session_id: 'runtime-1', title: 'Bot Chat' }))
      }
      return {}
    }
  })

  // No kickoff option: the click-path mint. The eager title write persists
  // the row, so NO intro turn fires — the user speaks first (ScottFive).
  assert.equal(await runtime.createCanonicalChat('ops'), 'stored-1')
  assert.deepEqual(events, [
    'session.list',
    'session.create',
    'session.title',
    'open:stored-1'
  ])
})

test('New Agent creation (kickoff: true) still sends the one intro turn', async () => {
  const events = []
  const runtime = loadCanonicalCreation({
    openSession: async id => events.push(`open:${id}`),
    request: async (method, params) => {
      events.push(method)
      if (method === 'session.create') return { stored_session_id: 'stored-1', session_id: 'runtime-1' }
      if (method === 'prompt.submit') {
        assert.equal(params.session_id, 'runtime-1')
      }
      return {}
    }
  })

  assert.equal(await runtime.createCanonicalChat('ops', { kickoff: true }), 'stored-1')
  assert.deepEqual(events, [
    'session.list',
    'session.create',
    'session.title',
    'open:stored-1',
    'prompt.submit'
  ])
})

test('compatibility: navigation retries after kickoff when eager title persistence is unavailable', async () => {
  const events = []
  let attempts = 0
  const runtime = loadCanonicalCreation({
    openSession: async id => {
      events.push(`open:${id}`)
      attempts += 1
      if (attempts === 1) throw new Error('stored row not persisted yet')
    },
    request: async method => {
      if (method === 'session.create') return { stored_session_id: 'stored-1', session_id: 'runtime-1' }
      if (method === 'session.title') throw new Error('unknown method')
      if (method === 'prompt.submit') events.push('kickoff:persisted')
      return {}
    }
  })

  assert.equal(await runtime.createCanonicalChat('ops'), 'stored-1')
  assert.deepEqual(events, ['open:stored-1', 'kickoff:persisted', 'open:stored-1'])
})

test('regression: a failed intro still returns the created registry row', async () => {
  const runtime = loadCanonicalCreation({
    openSession: async () => undefined,
    request: async method => {
      if (method === 'session.create') return { stored_session_id: 'new-bot-chat', session_id: 'rt-1' }
      if (method === 'prompt.submit') throw new Error('gateway timeout')
      return {}
    }
  })

  // The chat exists under the canonical title — the next click finds it by
  // NAME (the registry), so a failed kickoff can never orphan or fork it.
  assert.equal(await runtime.createCanonicalChat('newbie', { kickoff: true }), 'new-bot-chat')
})

test('a superseded roster open may create the canonical row but never navigate it', async () => {
  const opens = []
  let current = true
  const runtime = loadCanonicalCreation({
    openSession: async (id, options) => opens.push({ id, options }),
    request: async method => {
      if (method === 'session.create') {
        current = false
        return { stored_session_id: 'new-stored', session_id: 'new-runtime' }
      }
      return {}
    }
  })

  assert.equal(await runtime.createCanonicalChat('ops', { openingStillCurrent: () => current }), 'new-stored')
  assert.deepEqual(opens, [])
})

test('a newer same-bot open replaces the in-flight creation navigation guard', async () => {
  const opens = []
  let releaseCreate
  let markCreateStarted
  let firstCurrent = true
  const createStarted = new Promise(resolve => {
    markCreateStarted = resolve
  })
  const runtime = loadCanonicalCreation({
    openSession: async (id, options) => opens.push({ id, options }),
    request: async method => {
      if (method === 'session.create') {
        markCreateStarted()
        return new Promise(resolve => {
          releaseCreate = () => resolve({ stored_session_id: 'shared-stored', session_id: 'shared-runtime' })
        })
      }
      return {}
    }
  })

  const first = runtime.createCanonicalChat('ops', { openingStillCurrent: () => firstCurrent })
  await createStarted
  firstCurrent = false
  const second = runtime.createCanonicalChat('ops', { openingStillCurrent: () => true })
  releaseCreate()

  assert.equal(await first, 'shared-stored')
  assert.equal(await second, 'shared-stored')
  assert.equal(opens.length, 1, 'the newer current click owns the shared creation navigation')
})
