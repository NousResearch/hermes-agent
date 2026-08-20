import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// #24's guarantee — a VALID canonical pin is opened as-is, never replaced,
// and only an ACTUALLY-missing pin triggers recovery — used to be pinned
// against the old implementation's source shape (session.list rows[0]
// fallback). hermes-agent#88200 replaced that windowed, hidden-excluding
// lookup with the backend's precise preferred_session resolver, so the
// guarantee is now pinned as BEHAVIOR: what gets opened, what gets saved,
// and what never happens to a live pin.

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

  assert.notEqual(start, -1, 'canonical chat section is missing')
  assert.notEqual(end, -1, 'canonical chat section delimiter is missing')
  vm.runInNewContext(section, context, { filename: 'canonical-pin.js' })
  return { ...context.__open, saved, requests }
}

test('regression: a live pinned canonical chat is opened as-is, never replaced', async () => {
  const opened = []
  const runtime = loadOpenPath({
    openSession: async id => opened.push(id),
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

  assert.equal(await runtime.openBotCanonicalChat('ops', 'pin-1', null), 'pin-1')
  assert.deepEqual(opened, ['pin-1'], 'the pin itself is opened under the bot profile')
  assert.deepEqual(runtime.saved, [], 'a live pin is never rewritten')
  assert.equal(runtime.requests.some(r => r.method === 'session.create'), false,
    'a live pin never triggers a replacement chat')
})

test('regression: only an actually-missing pin triggers recovery', async () => {
  const runtime = loadOpenPath({
    openSession: async () => undefined,
    request: async method => {
      if (method === 'profiles.list') {
        return { profiles: [{ name: 'ops', preferred_session: null }] }
      }
      return {}
    }
  })

  // Definitively gone, but the roster still previews a live session —
  // recovery re-anchors on THAT session instead of minting a new chat.
  const history = { id: 'hist-1', title: 'Bot Chat', preview: 'p', last_active: 1 }
  assert.equal(await runtime.openBotCanonicalChat('ops', 'dead-pin', history), 'hist-1')
  assert.deepEqual(runtime.saved, [{ name: 'ops', patch: { chat: 'hist-1' } }])
})
