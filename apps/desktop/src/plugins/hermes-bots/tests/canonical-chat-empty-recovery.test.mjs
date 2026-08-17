import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadCanonicalRecovery({ openSession, request }) {
  const start = source.indexOf('const canonicalCreations = new Map()')
  const end = source.indexOf('function displayName(', start)
  const saved = []
  const context = {
    host: { openSession, request },
    saveBotMeta: (name, patch) => saved.push({ name, patch }),
    $hideBotChats: { get: () => false },
    window: { setTimeout: callback => callback() }
  }
  const section = source.slice(start, end).concat('\nglobalThis.__canonical = { openBotCanonicalChat };\n')

  assert.notEqual(start, -1, 'canonical chat section is missing')
  assert.notEqual(end, -1, 'canonical chat section delimiter is missing')
  vm.runInNewContext(section, context, { filename: 'canonical-recovery.js' })
  return { ...context.__canonical, saved }
}

test('regression: an empty session list clears a stale pin and creates a replacement', async () => {
  const opened = []
  const runtime = loadCanonicalRecovery({
    openSession: async id => opened.push(id),
    request: async method => {
      if (method === 'session.list') return { sessions: [] }
      if (method === 'session.create') return { stored_session_id: 'replacement', session_id: 'replacement-runtime' }
      return {}
    }
  })

  assert.equal(await runtime.openBotCanonicalChat('ops', 'stale-pin'), 'replacement')
  assert.deepEqual(opened, ['replacement'])
  assert.deepEqual(JSON.parse(JSON.stringify(runtime.saved)), [
    { name: 'ops', patch: { chat: null } },
    { name: 'ops', patch: { chat: 'replacement' } }
  ])
})

test('regression: an unpinned bot adopts an existing Bot Chat instead of creating another', async () => {
  const opened = []
  const requests = []
  const runtime = loadCanonicalRecovery({
    openSession: async id => opened.push(id),
    request: async method => {
      requests.push(method)
      if (method === 'session.list') {
        return {
          sessions: [
            { id: 'latest-scratch', title: 'Scratch chat' },
            { id: 'existing-canonical', title: 'Bot Chat' }
          ]
        }
      }
      throw new Error(`unexpected ${method}`)
    }
  })

  assert.equal(await runtime.openBotCanonicalChat('ops', null), 'existing-canonical')
  assert.deepEqual(opened, ['existing-canonical'])
  assert.deepEqual(requests, ['session.list'])
  assert.deepEqual(JSON.parse(JSON.stringify(runtime.saved)), [{ name: 'ops', patch: { chat: 'existing-canonical' } }])
})

test('regression: a missing pin reuses a Bot Chat instead of the newest session', async () => {
  const opened = []
  const runtime = loadCanonicalRecovery({
    openSession: async id => opened.push(id),
    request: async method => {
      if (method === 'session.list') {
        return {
          sessions: [
            { id: 'scratch-from-cli', title: 'scratch' },
            { id: 'the-real-bot-chat', title: 'Bot Chat' }
          ]
        }
      }
      if (method === 'session.create') throw new Error('must not create')
      return {}
    }
  })

  assert.equal(await runtime.openBotCanonicalChat('ops', 'old-pin'), 'the-real-bot-chat')
  assert.deepEqual(opened, ['the-real-bot-chat'])
  assert.deepEqual(JSON.parse(JSON.stringify(runtime.saved)), [
    { name: 'ops', patch: { chat: 'the-real-bot-chat' } }
  ])
})

test('regression: a pin not in the last page is opened as-is when no Bot Chat title exists', async () => {
  const opened = []
  const runtime = loadCanonicalRecovery({
    openSession: async id => opened.push(id),
    request: async (method, params) => {
      if (method === 'session.list') {
        assert.equal(params.limit, 100)
        return { sessions: [{ id: 'recent-0', title: 'scratch' }] }
      }
      if (method === 'session.create') throw new Error('must not create')
      return {}
    }
  })

  assert.equal(await runtime.openBotCanonicalChat('ops', 'old-pin-outside-page'), 'old-pin-outside-page')
  assert.deepEqual(opened, ['old-pin-outside-page'])
  assert.equal(runtime.saved.length, 0)
})
