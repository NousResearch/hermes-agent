import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// #93602: sub-profile bot silent in a group room. A member turn is a
// session-scoped RPC sequence (resume → attach → prompt.submit → poll) issued
// with the runtime id its FIRST rpc minted, but every requestForBot call rides
// its own request-scoped socket lease — at refcount 0 the leased secondary is
// disposed, the gateway detaches the runtime session on WS disconnect, the
// orphan reaper frees it, and the next RPC dies 4001 "not in memory" with no
// reply in the room. The fix holds a retained per-turn lease
// (host.retainProfile) across the whole sequence and adds a one-shot
// catch-and-retry on prompt.submit that re-resumes via the STORED id.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

/** Harness mirroring group-chat-attachments.test.mjs, extended with a
 *  refcounted host.requestProfile/host.retainProfile pair so socket lease
 *  lifetimes are assertable. */
function load({ failFirstSubmitWith = null, failEverySubmitWith = null, reply = 'hello from turn' } = {}) {
  const values = new Map()
  const atom = initial => {
    const slot = { get: () => values.get(slot), set: value => values.set(slot, value) }
    values.set(slot, initial)
    return slot
  }

  const sessions = new Map()
  const runtimeToStored = new Map()
  const titleToStored = new Map()
  let sessionSequence = 0
  let submits = 0
  const rpcLog = []

  // Mock socket-pool refcount for the member's route: every RPC takes a
  // per-request lease (acquire → handle → release), exactly like
  // requestGatewayForAgent in store/gateway.ts. `disposals` counts every time
  // the refcount hits 0 — each one is a socket close that reaps the runtime
  // session in production.
  let refcount = 0
  let disposals = 0
  const timeline = []

  const acquire = () => {
    refcount += 1
  }
  const release = () => {
    refcount -= 1
    if (refcount === 0) {
      disposals += 1
    }
  }

  const resolveSession = (profile, target) =>
    (stored => (stored ? sessions.get(stored) : null))(
      runtimeToStored.get(target) || (sessions.has(target) ? target : titleToStored.get(`${profile}::${target}`))
    )

  const handle = async (method, params) => {
    if (method === 'session.create') {
      sessionSequence += 1
      const stored = `sid-${sessionSequence}`
      const runtime = `rt-${sessionSequence}`
      const session = { stored, runtime, profile: params.profile, title: params.title, messages: [] }
      sessions.set(stored, session)
      runtimeToStored.set(runtime, stored)
      titleToStored.set(`${params.profile}::${params.title}`, stored)
      return { session_id: runtime, stored_session_id: stored, message_count: 0, messages: [] }
    }
    if (method === 'session.resume') {
      const session = resolveSession(params.profile, params.session_id)
      if (!session) {
        const err = new Error(`session not found: ${params.session_id}`)
        err.code = 4007
        throw err
      }
      // Every resume mints a FRESH runtime id — the stored id is the durable
      // identity, mirroring the gateway's resume contract.
      sessionSequence += 1
      const runtime = `rt-${sessionSequence}`
      session.runtime = runtime
      runtimeToStored.set(runtime, session.stored)
      return {
        session_id: runtime,
        session_key: session.stored,
        message_count: session.messages.length,
        messages: params.omit_messages ? [] : [...session.messages],
        inflight: false,
        running: false
      }
    }
    if (method === 'image.attach_bytes' || method === 'pdf.attach' || method === 'file.attach') {
      const session = resolveSession(null, params.session_id)
      if (!session) {
        const err = new Error(`session-scoped RPC rejected: ${params.session_id} not in memory`)
        err.code = 4001
        throw err
      }
      return { attached: true }
    }
    if (method === 'prompt.submit') {
      submits += 1
      if (failEverySubmitWith) {
        throw failEverySubmitWith
      }
      if (submits === 1 && failFirstSubmitWith) {
        throw failFirstSubmitWith
      }
      const session = resolveSession(null, params.session_id)
      if (!session) {
        const err = new Error(`session-scoped RPC rejected: ${params.session_id} not in memory`)
        err.code = 4001
        throw err
      }
      session.messages.push({ role: 'user', content: params.text })
      session.messages.push({ role: 'assistant', content: reply })
      return {}
    }
    return {}
  }

  const context = {
    atom,
    setTimeout: fn => {
      fn()
      return 0
    },
    clearTimeout: () => undefined,
    Date,
    console,
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: {
      request: async (method, params) => handle(method, params),
      requestProfile: async (route, method, params) => {
        acquire()
        try {
          return await handle(method, params)
        } finally {
          release()
          rpcLog.push({ method, refcountAfter: refcount })
          timeline.push(method)
        }
      },
      retainProfile: async () => {
        timeline.push('retain')
        acquire()
        let released = false
        return () => {
          if (!released) {
            released = true
            timeline.push('release')
            release()
          }
        }
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
      '\nglobalThis.__lease = { runGroupChatMemberTurn, submitGroupTurnPrompt, isSessionGoneError, $groupChats };\n'
    )
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  context.plugin.register({
    storage: { get: () => null, set: () => undefined },
    register: () => undefined
  })
  return {
    ...context.__lease,
    context,
    rpcLog,
    timeline,
    stats: () => ({ refcount, disposals, submits })
  }
}

const LOCAL_MEMBER = { name: 'helper', title: '' }
const ROUTED_MEMBER = { name: 'helper', connectionId: 'mini', remoteSource: true }
const IMG = { name: 'shot.png', data: 'data:image/png;base64,iVBORw0KGgo=' }

test('isSessionGoneError: 4001 and "not in memory" are recoverable, 4007 is not', () => {
  const gc = load()
  const gone = new Error('x')
  gone.code = 4001
  assert.equal(gc.isSessionGoneError(gone), true)
  assert.equal(gc.isSessionGoneError(new Error('session_id=rt-1 not in memory')), true)
  const missing = new Error('session not found')
  missing.code = 4007
  assert.equal(gc.isSessionGoneError(missing), false)
  assert.equal(gc.isSessionGoneError(null), false)
  assert.equal(gc.isSessionGoneError(new Error('network blip')), false)
})

test('a 4001 on the first prompt.submit recovers via session.resume on the STORED id and delivers', async () => {
  const reaped = new Error("session-scoped RPC rejected: session_id='rt-1' not in memory")
  reaped.code = 4001
  const gc = load({ failFirstSubmitWith: reaped, reply: 'recovered reply' })

  const reply = await gc.runGroupChatMemberTurn('Room', LOCAL_MEMBER, 'hi there', 't1', [])

  assert.equal(reply, 'recovered reply')
  // One failed submit + exactly one retry — never more.
  assert.equal(gc.stats().submits, 2)
  // The recovery re-resumed the durable stored id, not the dead runtime id.
  const room = gc.$groupChats.get().Room
  assert.ok(room.sessions.helper, 'stored session id survives in the room record')
})

test('a persistent non-4001 submit failure is NOT retried and still surfaces', async () => {
  const fatal = new Error('backend exploded')
  const gc = load({ failEverySubmitWith: fatal })

  await assert.rejects(() => gc.runGroupChatMemberTurn('Room', LOCAL_MEMBER, 'hi', 't1', []), /backend exploded/)
  assert.equal(gc.stats().submits, 1)
})

test('the per-turn lease is acquired before any session RPC and held across attach+submit', async () => {
  const gc = load({ reply: 'routed reply' })

  const reply = await gc.runGroupChatMemberTurn('Room', ROUTED_MEMBER, 'look at this', 't1', [IMG])

  assert.equal(reply, 'routed reply')
  // The retain landed before the first session-scoped RPC on the route.
  assert.equal(gc.timeline[0], 'retain')
  // The socket was NEVER disposed mid-turn: after every per-request lease
  // released, the turn lease still held the refcount above zero.
  for (const rpc of gc.rpcLog) {
    assert.ok(rpc.refcountAfter >= 1, `${rpc.method} left refcount ${rpc.refcountAfter} — socket disposed mid-turn`)
  }
  // Exactly one disposal, and only via the turn lease's own release at the end.
  assert.equal(gc.stats().disposals, 1)
  assert.equal(gc.timeline[gc.timeline.length - 1], 'release')
})

test('the per-turn lease is released after the turn — refcount returns to zero', async () => {
  const gc = load({ reply: 'done' })

  await gc.runGroupChatMemberTurn('Room', ROUTED_MEMBER, 'hi', 't1', [])

  assert.equal(gc.stats().refcount, 0)
  assert.equal(gc.stats().disposals, 1)
})

test('the per-turn lease is released even when the turn fails', async () => {
  const fatal = new Error('backend exploded')
  const gc = load({ failEverySubmitWith: fatal })

  await assert.rejects(() => gc.runGroupChatMemberTurn('Room', ROUTED_MEMBER, 'hi', 't1', []))

  assert.equal(gc.stats().refcount, 0)
})

test('hosts without retainProfile still run the turn (feature detection)', async () => {
  const gc = load({ reply: 'legacy ok' })
  delete gc.context.host.retainProfile

  const reply = await gc.runGroupChatMemberTurn('Room', ROUTED_MEMBER, 'hi', 't1', [])

  assert.equal(reply, 'legacy ok')
})
