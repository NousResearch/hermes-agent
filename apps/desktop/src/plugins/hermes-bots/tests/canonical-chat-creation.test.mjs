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
        assert.deepEqual(params, { session_id: 'runtime-1', title: 'Bot Chat' })
      }
      return {}
    }
  })

  assert.equal(await runtime.createCanonicalChat('ops'), 'stored-1')
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
  assert.equal(await runtime.createCanonicalChat('newbie'), 'new-bot-chat')
})
