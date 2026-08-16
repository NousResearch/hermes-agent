import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadSoulHelpers({ serverInjects = false } = {}) {
  const start = source.indexOf('function botHandle(name) {')
  const end = source.indexOf('// ── human-readable row helpers', start)
  assert.notEqual(start, -1, 'botHandle is missing')
  assert.notEqual(end, -1, 'soul-helper section delimiter is missing')
  const context = { serverInjectsProtocol: serverInjects }
  vm.runInNewContext(
    `${source.slice(start, end)}\nObject.assign(globalThis, { botHandle, hasMessagingProtocol, ensureMessagingProtocol, composeSoul, messagingProtocolSection })`,
    context
  )
  return context
}

const EXISTING_SOUL = `# Main agent

I am the default profile on this machine.

## Notes
- Execute directly.
`

test('generated protocol uses hermes profile list, not the invalid profiles plural', () => {
  assert.match(source, /run `hermes profile list` for the LIVE/)
  assert.doesNotMatch(source, /run `hermes profiles list` for the LIVE/)
})

test('pre-existing custom SOUL is detected as missing the protocol', () => {
  const { hasMessagingProtocol } = loadSoulHelpers()
  assert.equal(hasMessagingProtocol(EXISTING_SOUL), false)
  assert.equal(hasMessagingProtocol(''), false)
  assert.equal(hasMessagingProtocol('# Bot\n\n## Messaging other agents\n\nYou work alongside'), true)
})

test('ensureMessagingProtocol appends once and does not duplicate', () => {
  const { ensureMessagingProtocol, hasMessagingProtocol } = loadSoulHelpers()
  const roster = [
    { name: 'default', description: 'main agent' },
    { name: 'researcher', description: 'research specialist' }
  ]
  const once = ensureMessagingProtocol(EXISTING_SOUL, 'default', roster)
  assert.equal(hasMessagingProtocol(once), true)
  assert.match(once, /I am the default profile on this machine/)
  assert.match(once, /`researcher` — research specialist/)
  assert.match(once, /@hermes/)
  assert.doesNotMatch(once, /@default/)

  const twice = ensureMessagingProtocol(once, 'default', roster)
  assert.equal(twice, once.trim())
  assert.equal(twice.split('## Messaging other agents').length, 2)
})

test('composeSoul does not double-append when custom SOUL already has the protocol', () => {
  const { composeSoul } = loadSoulHelpers()
  const roster = [{ name: 'researcher' }, { name: 'default' }]
  const withProtocol = composeSoul({
    name: 'researcher',
    title: 'Researcher',
    description: 'literature review',
    roster,
    customSoul: ''
  })
  const cloned = composeSoul({
    name: 'researcher',
    roster,
    customSoul: withProtocol
  })
  assert.equal(cloned.split('## Messaging other agents').length, 2)
})

test('backfill is one-shot: describe-only when protocol exists, configure when missing', async () => {
  const start = source.indexOf('const soulProtocolChecked = new Set()')
  const end = source.indexOf('/** SOUL.md for a new bot:', start)
  assert.notEqual(start, -1)
  assert.notEqual(end, -1)

  const calls = []
  const context = {
    serverInjectsProtocol: false,
    soulProtocolChecked: new Set(),
    soulProtocolInflight: new Set(),
    hasMessagingProtocol: soul => /## Messaging other agents/.test(soul || ''),
    ensureMessagingProtocol: soul => `${soul}\n\n## Messaging other agents\n`,
    host: {
      request: async (method, payload) => {
        calls.push({ method, payload })
        if (method === 'profiles.describe') {
          if (payload.name === 'researcher') return { soul: '# Researcher\n\n## Messaging other agents\n' }
          return { soul: EXISTING_SOUL }
        }
        return { ok: true }
      }
    }
  }

  vm.runInNewContext(
    `${source.slice(start, end)}\nbackfillMessagingProtocol(roster)`,
    { ...context, roster: [{ name: 'default' }, { name: 'researcher' }] }
  )

  await new Promise(resolve => setTimeout(resolve, 0))
  await new Promise(resolve => setTimeout(resolve, 0))

  const describes = calls.filter(c => c.method === 'profiles.describe').map(c => c.payload.name)
  const configures = calls.filter(c => c.method === 'profiles.configure')
  assert.deepEqual(describes.sort(), ['default', 'researcher'])
  assert.equal(configures.length, 1)
  assert.equal(configures[0].payload.name, 'default')
  assert.match(configures[0].payload.soul, /## Messaging other agents/)
})

test('backend bot_mode_protocol capability suppresses every SOUL protocol write', async () => {
  // ensureMessagingProtocol becomes identity — custom SOULs stay untouched.
  const helpers = loadSoulHelpers({ serverInjects: true })
  assert.equal(helpers.ensureMessagingProtocol(EXISTING_SOUL, 'default', []), EXISTING_SOUL.trim())

  // backfill never issues a single RPC.
  const start = source.indexOf('const soulProtocolChecked = new Set()')
  const end = source.indexOf('/** SOUL.md for a new bot:', start)
  const calls = []
  vm.runInNewContext(
    `${source.slice(start, end)}\nbackfillMessagingProtocol(roster)`,
    {
      serverInjectsProtocol: true,
      soulProtocolChecked: new Set(),
      soulProtocolInflight: new Set(),
      hasMessagingProtocol: () => false,
      ensureMessagingProtocol: soul => soul,
      host: { request: async (method, payload) => (calls.push({ method, payload }), { ok: true }) },
      roster: [{ name: 'default' }, { name: 'researcher' }]
    }
  )
  await new Promise(resolve => setTimeout(resolve, 0))
  assert.equal(calls.length, 0)

  // composeSoul's generated-identity path also skips the section.
  const generated = helpers.composeSoul({ name: 'newbot', title: 'T', description: 'D', roster: [], customSoul: '' })
  assert.doesNotMatch(generated, /## Messaging other agents/)
})
