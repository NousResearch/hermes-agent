import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadCanonicalRecovery({ openSession, request }) {
  const start = source.indexOf('const canonicalCreations = new Map()')
  const end = source.indexOf('function displayName(', start)
  const saved = []
  const requests = []
  const context = {
    host: {
      openSession,
      request: async (method, params) => {
        requests.push(method)
        return request(method, params)
      }
    },
    saveBotMeta: (name, patch) => saved.push({ name, patch }),
    $hideBotChats: { get: () => false },
    window: { setTimeout: callback => callback() }
  }
  const section = source.slice(start, end).concat('\nglobalThis.__canonical = { openBotCanonicalChat };\n')

  assert.notEqual(start, -1, 'canonical chat section is missing')
  assert.notEqual(end, -1, 'canonical chat section delimiter is missing')
  vm.runInNewContext(section, context, { filename: 'canonical-recovery.js' })
  return { ...context.__canonical, saved, requests }
}

test('regression: a definitively-gone pin with no history clears and creates a replacement', async () => {
  // New contract (hermes-agent#88200): the pin is verified through the
  // backend's precise preferred_session resolver — NOT a paginated,
  // hidden-excluding session.list window. preferred_session=null is the
  // definitive "this session is gone"; with no previewed history to
  // re-anchor on, recovery clears the pin and creates a fresh chat.
  const opened = []
  const runtime = loadCanonicalRecovery({
    openSession: async id => opened.push(id),
    request: async method => {
      if (method === 'profiles.list') return { profiles: [{ name: 'ops', preferred_session: null }] }
      if (method === 'session.create') return { stored_session_id: 'replacement', session_id: 'replacement-runtime' }
      return {}
    }
  })

  assert.equal(await runtime.openBotCanonicalChat('ops', 'stale-pin', null), 'replacement')
  assert.deepEqual(opened, ['replacement'])
  assert.deepEqual(JSON.parse(JSON.stringify(runtime.saved)), [
    { name: 'ops', patch: { chat: null } },
    { name: 'ops', patch: { chat: 'replacement' } }
  ])
})

test('regression: an unpinned bot adopts its previewed chat instead of creating another', async () => {
  // A CLI/A2A exchange can create the canonical chat before the desktop
  // saves ui_meta.chat. Grandfathering adopts the session the row already
  // previews (the roster's last_session) rather than minting a new one.
  const opened = []
  const runtime = loadCanonicalRecovery({
    openSession: async id => opened.push(id),
    request: async method => {
      if (method === 'session.create') throw new Error('must not create')
      return {}
    }
  })

  const history = { id: 'existing-canonical', title: 'Bot Chat', preview: 'hey', last_active: 5 }
  assert.equal(await runtime.openBotCanonicalChat('ops', null, history), 'existing-canonical')
  assert.deepEqual(opened, ['existing-canonical'])
  assert.deepEqual(JSON.parse(JSON.stringify(runtime.saved)), [
    { name: 'ops', patch: { chat: 'existing-canonical' } }
  ])
})

test('regression: a dead pin re-anchors on the previewed chat instead of the newest session', async () => {
  // hermes-agent#88146: recovery must never steal an unrelated scratch
  // session. A pin the backend reports definitively gone re-anchors on the
  // previewed history row when one exists — session.create never fires.
  const opened = []
  const runtime = loadCanonicalRecovery({
    openSession: async id => opened.push(id),
    request: async method => {
      if (method === 'profiles.list') return { profiles: [{ name: 'ops', preferred_session: null }] }
      if (method === 'session.create') throw new Error('must not create')
      return {}
    }
  })

  const history = { id: 'the-real-bot-chat', title: 'Bot Chat', preview: 'p', last_active: 9 }
  assert.equal(await runtime.openBotCanonicalChat('ops', 'old-pin', history), 'the-real-bot-chat')
  assert.deepEqual(opened, ['the-real-bot-chat'])
  assert.deepEqual(JSON.parse(JSON.stringify(runtime.saved)), [
    { name: 'ops', patch: { chat: 'the-real-bot-chat' } }
  ])
})

test('regression: an inconclusive lookup opens the stored pin as-is and never rewrites it', async () => {
  // hermes-agent#88146: an older backend (profiles.list without the
  // preferred_session_ids param) or a transient hiccup is NOT proof the pin
  // is gone. The pin is opened as-is; nothing is saved, nothing is created.
  const opened = []
  const runtime = loadCanonicalRecovery({
    openSession: async id => opened.push(id),
    request: async method => {
      if (method === 'profiles.list') return { profiles: [{ name: 'ops' }] }
      if (method === 'session.create') throw new Error('must not create')
      return {}
    }
  })

  assert.equal(await runtime.openBotCanonicalChat('ops', 'old-pin-outside-page', null), 'old-pin-outside-page')
  assert.deepEqual(opened, ['old-pin-outside-page'])
  assert.equal(runtime.saved.length, 0)
})
