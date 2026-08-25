import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// The @mention middleware is IDENTIFICATION-ONLY (Aug 2026 redesign): it
// resolves the user's @tags against the live roster and annotates the draft
// with who they refer to. It never delivers anything — the agent owns
// messaging via its Bot-Chat message_agent tool, so there is exactly one
// send path, no renderer-side shellout instructions, and no verbatim
// forwarding of the user's text (the class behind #91397/#91304/#91339).

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function load({
  activeProfile = 'research',
  focusedProfile = activeProfile,
  profiles = ['research', 'ops'],
  title = null,
  unionProfiles = null,
  requestProfileSpy = null
} = {}) {
  const values = new Map()
  const atom = initial => {
    const slot = { get: () => values.get(slot), set: value => values.set(slot, value) }
    values.set(slot, initial)
    return slot
  }
  const context = {
    atom,
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: {
      request: async method => {
        if (method === 'profiles.list') {
          return {
            profiles: profiles.map(profile =>
              typeof profile === 'string' ? { name: profile } : profile
            )
          }
        }
        return {}
      },
      ...(requestProfileSpy ? { requestProfile: requestProfileSpy } : {}),
      state: {
        profile: { get: () => activeProfile, listen: () => undefined },
        focusedSessionProfile: { get: () => focusedProfile, listen: () => undefined },
        connectionId: { get: () => 'local', listen: () => undefined },
        gateway: { listen: () => undefined }
      }
    },
    ...(unionProfiles
      ? { queryClient: { getQueryData: () => ({ profiles: unionProfiles }) } }
      : {})
  }
  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat('\nglobalThis.__mention = { $botMeta };\n')
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  context.__mention.$botMeta.set(title ? { [activeProfile]: { title } } : {})

  const registered = []
  context.plugin.register({ storage: { get: () => null }, register: entry => registered.push(entry) })
  const middleware = registered.find(entry => entry.id === 'mention-middleware')
  assert.ok(middleware, 'mention middleware did not register')
  return { handler: middleware.data.handler }
}

test('identification: a local mention annotates who the user means', async () => {
  const { handler } = load()
  const result = await handler({ text: 'please @ops review the diff' })
  assert.match(result.text, /@mentions resolved from the Bot Mode roster/)
  assert.match(result.text, /@ops = agent profile "ops"/)
  assert.match(result.text, /message_agent/)
})

test('containment: the note never teaches a shellout and never forwards a command', async () => {
  const { handler } = load()
  const result = await handler({ text: 'ask @ops to summarize' })
  assert.doesNotMatch(result.text, /hermes -p/)
  assert.doesNotMatch(result.text, /terminal call/i)
  assert.doesNotMatch(result.text, /background=true/)
})

test('containment: the note tells the agent to compose, never forward verbatim', async () => {
  const { handler } = load()
  const result = await handler({ text: '@ops handle this' })
  assert.match(result.text, /compose your own message/i)
  assert.match(result.text, /never forward/i)
})

test('security: a poisoned bot title stays inert prose (no shell context exists)', async () => {
  const title = 'Evil" ; touch /tmp/pwned ; echo "$(touch /tmp/pwned2)"'
  const { handler } = load({
    activeProfile: 'ops',
    focusedProfile: 'ops',
    profiles: [{ name: 'ops' }, { name: 'research', display_name: title }]
  })
  const result = await handler({ text: 'ping @research please' })
  // The note is plain prose fed to the model — there is no command to break
  // out of. The only invariant left: no hermes command is ever emitted.
  assert.doesNotMatch(result.text, /`hermes/)
})

test('remote mentions: identified with their device, never delivered by the renderer', async () => {
  const delivered = []
  const { handler } = load({
    activeProfile: 'default',
    focusedProfile: 'default',
    unionProfiles: [
      { name: 'default', connectionId: 'local' },
      {
        name: 'dixie',
        connectionId: 'mac-mini',
        connectionLabel: 'Mac Mini',
        handle: 'dixie',
        remoteSource: true
      }
    ],
    requestProfileSpy: async (...args) => {
      delivered.push(args)
      return {}
    }
  })

  const result = await handler({ text: '@dixie what is the disk space?' })
  assert.match(result.text, /@dixie = agent profile "dixie"/)
  assert.match(result.text, /on Mac Mini/)
  // The renderer must NOT deliver: no requestProfile traffic at all.
  await new Promise(resolve => setTimeout(resolve, 50))
  assert.equal(delivered.length, 0, 'middleware must never deliver over Connections')
})

test('unknown @ and emails pass through untouched', async () => {
  const { handler } = load()
  const untouched = 'mail user@example.com and ping @nosuchbot'
  const result = await handler({ text: untouched })
  assert.equal(result.text, untouched)
})

test('source contract: the delivery machinery is gone from plugin.js', () => {
  assert.doesNotMatch(pluginSource, /deliverRemoteRosterMentions/)
  assert.doesNotMatch(pluginSource, /pollRemoteDmReply/)
  assert.doesNotMatch(pluginSource, /ensureRemoteCanonicalChat/)
  assert.doesNotMatch(pluginSource, /REMOTE_DM_TIMEOUT_MS/)
  // The middleware must not know how to build a bot-to-bot hermes command.
  assert.doesNotMatch(pluginSource, /\[@mention handoff/)
  assert.doesNotMatch(pluginSource, /Desktop is delivering/)
})
