import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

/** Harness mirroring group-chat.test.mjs, extended to record image.attach_bytes
 *  calls so attachment staging order and fan-out are assertable. */
function load(turnScript) {
  const values = new Map()
  const atom = initial => {
    const slot = { get: () => values.get(slot), set: value => values.set(slot, value) }
    values.set(slot, initial)
    return slot
  }
  const calls = []
  const attaches = []
  const sessions = new Map()
  const runtimeToStored = new Map()
  const titleToStored = new Map()
  let sessionSequence = 0

  const resolveSession = (profile, target) => {
    const stored = runtimeToStored.get(target) || (sessions.has(target) ? target : titleToStored.get(`${profile}::${target}`))
    return stored ? sessions.get(stored) : null
  }
  const context = {
    atom,
    setTimeout: fn => {
      fn()
      return 0
    },
    clearTimeout: () => undefined,
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: {
      request: async (method, params) => {
        if (method === 'session.create') {
          sessionSequence += 1
          const stored = `sid-${params.profile}-${sessionSequence}`
          const runtime = `rt-${params.profile}-${sessionSequence}`
          const session = { stored, runtime, profile: params.profile, title: params.title, messages: [] }
          sessions.set(stored, session)
          runtimeToStored.set(runtime, stored)
          titleToStored.set(`${params.profile}::${params.title}`, stored)
          return { session_id: runtime, stored_session_id: stored, message_count: 0, messages: [] }
        }
        if (method === 'session.resume') {
          const session = resolveSession(params.profile, params.session_id)
          if (!session) {
            throw new Error(`session not found: ${params.session_id}`)
          }
          return {
            session_id: session.runtime,
            session_key: session.stored,
            message_count: session.messages.length,
            messages: [...session.messages],
            inflight: false,
            running: false
          }
        }
        if (method === 'image.attach_bytes') {
          const session = resolveSession(null, params.session_id)
          attaches.push({
            profile: session ? session.profile : null,
            runtime: params.session_id,
            filename: params.filename,
            data: params.content_base64,
            order: calls.length // how many prompt.submits happened before this attach
          })
          return { attached: true }
        }
        if (method === 'prompt.submit') {
          const session = resolveSession(null, params.session_id)
          if (!session) {
            throw new Error(`runtime session not found: ${params.session_id}`)
          }
          session.messages.push({ role: 'user', content: params.text })
          calls.push({ profile: session.profile, prompt: params.text, runtime: session.runtime })
          const reply = turnScript(session.profile, params.text, calls.length, session)
          session.messages.push({ role: 'assistant', content: reply })
          return {}
        }
        return {}
      },
      state: { profile: { get: () => 'default', listen: () => undefined }, gateway: { listen: () => undefined } },
      notify: () => undefined,
      notifyError: () => undefined
    }
  }
  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat(
      '\nglobalThis.__gca = { sendToGroupChat, formatGroupChatLine, $groupChats };\n'
    )
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  context.plugin.register({
    storage: { get: () => null, set: () => undefined },
    register: () => undefined
  })
  return { ...context.__gca, calls, attaches, sessions }
}

const MEMBERS = [{ name: 'research', title: '' }, { name: 'builder', title: '' }]
const IMG = { name: 'screenshot.png', data: 'data:image/png;base64,iVBORw0KGgo=' }

async function drain(gc, group) {
  for (let i = 0; i < 200 && (gc.$groupChats.get()[group] || {}).running; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }
}

test('attachments are staged into EVERY responding member session before its submit', async () => {
  const gc = load(() => '(pass)')

  gc.sendToGroupChat('Vision', MEMBERS, 'what does this show?', null, [IMG])
  await drain(gc, 'Vision')

  // Both members got the image, on their own per-group sessions.
  assert.equal(gc.attaches.length, 2)
  assert.deepEqual(gc.attaches.map(a => a.profile).sort(), ['builder', 'research'])
  for (const attach of gc.attaches) {
    assert.equal(attach.filename, 'screenshot.png')
    assert.equal(attach.data, IMG.data)
  }
  // Each attach landed before that member's prompt.submit: the first attach
  // happens before any submit, the second before the second submit.
  assert.equal(gc.attaches[0].order, 0)
  assert.ok(gc.attaches[1].order <= 1)
  assert.equal(gc.calls.length, 2)
})

test('mention routing scopes attachments: only the mentioned member receives the image', async () => {
  const gc = load(() => '(pass)')

  gc.sendToGroupChat('Scoped', MEMBERS, '@builder look at this', null, [IMG])
  await drain(gc, 'Scoped')

  assert.deepEqual(gc.attaches.map(a => a.profile), ['builder'])
})

test('image-only send works and the room entry carries the attachment', async () => {
  const gc = load(() => '(pass)')

  const minted = gc.sendToGroupChat('Silent', MEMBERS, '', null, [IMG])
  await drain(gc, 'Silent')

  assert.ok(minted)
  const log = gc.$groupChats.get().Silent.log
  assert.equal(log.length, 1)
  assert.equal(log[0].images.length, 1)
  assert.equal(log[0].images[0].name, 'screenshot.png')
  // Members still got prompted (with the image staged).
  assert.equal(gc.attaches.length, 2)
  assert.equal(gc.calls.length, 2)
})

test('a text-only send stages nothing and old entries never re-attach on later turns', async () => {
  const gc = load(() => '(pass)')

  gc.sendToGroupChat('Plain', MEMBERS, 'with image', null, [IMG])
  await drain(gc, 'Plain')
  const afterFirst = gc.attaches.length
  assert.equal(afterFirst, 2)

  // Second, text-only send: the image already sits behind every member's
  // watermark, so no new attach calls may happen.
  gc.sendToGroupChat('Plain', MEMBERS, 'plain follow-up')
  await drain(gc, 'Plain')
  assert.equal(gc.attaches.length, afterFirst)
})

test('formatGroupChatLine names attachments for user and member entries', () => {
  const gc = load(() => '(pass)')

  const userLine = gc.formatGroupChatLine(
    { from: { kind: 'user', name: 'You' }, text: 'see attached', images: [IMG] },
    'research'
  )
  assert.equal(userLine, 'You (user): see attached [attached image: screenshot.png]')

  const noImages = gc.formatGroupChatLine({ from: { kind: 'user', name: 'You' }, text: 'plain' }, 'research')
  assert.equal(noImages, 'You (user): plain')

  const memberLine = gc.formatGroupChatLine(
    { from: { kind: 'member', name: 'builder' }, text: 'made this', images: [{ data: 'data:image/png;base64,x' }] },
    'research'
  )
  assert.equal(memberLine, 'builder: made this [attached image: image]')
})

test('an invalid attachment is skipped and the member turn still runs text-only', async () => {
  const gc = load(() => '(pass)')

  // runGroupChatMemberTurn skips entries without a string data field.
  gc.sendToGroupChat('Degraded', MEMBERS, 'look', null, [{ name: 'bad', data: null }])
  await drain(gc, 'Degraded')

  assert.equal(gc.attaches.length, 0) // invalid image skipped, no attach attempted
  assert.equal(gc.calls.length, 2) // both turns still ran
})
