import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// #91868/#94569: a REAL stop path for group-chat rounds. Before
// stopGroupThread, the loop's only cancellation primitives were the epoch
// bump (checked at member boundaries only) and #93129 holds (skip FUTURE
// turns) — the plugin issued ZERO session.interrupt RPCs, so "stop" meant
// "wait for the in-flight member to finish its whole model turn". The
// primitive bumps the epoch, holds every member, interrupts the member ON
// TURN, and the runGroupChatMemberTurnLeased poll loop abandons a turn whose
// epoch went stale while its member is held.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

/** Harness mirroring group-turn-lease.test.mjs: run plugin.js in a vm with a
 *  scripted gateway. `busyPolls` makes session.resume report the member as
 *  inflight for the first N polls, so a stop can land mid-turn. */
function load({ reply = 'long answer', busyPolls = 0, onResumePoll = null } = {}) {
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
  const rpcLog = []
  let resumePolls = 0

  const resolveSession = (profile, target) =>
    (stored => (stored ? sessions.get(stored) : null))(
      runtimeToStored.get(target) || (sessions.has(target) ? target : titleToStored.get(`${profile}::${target}`))
    )

  const handle = async (method, params) => {
    rpcLog.push({ method, params })

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

      sessionSequence += 1
      const runtime = `rt-${sessionSequence}`
      session.runtime = runtime
      runtimeToStored.set(runtime, session.stored)

      // Post-submit polls: stay "busy" for the first `busyPolls` polls so a
      // stop can land mid-turn, then settle. `onResumePoll` lets a test fire
      // the stop from inside the poll cadence.
      const submitted = session.messages.length > 0
      let busy = false

      if (submitted) {
        resumePolls += 1
        busy = resumePolls <= busyPolls

        if (typeof onResumePoll === 'function') {
          onResumePoll(resumePolls)
        }
      }

      return {
        session_id: runtime,
        session_key: session.stored,
        message_count: busy ? 0 : session.messages.length,
        messages: busy ? [] : [...session.messages],
        inflight: busy,
        running: false
      }
    }

    if (method === 'prompt.submit') {
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

    if (method === 'session.interrupt') {
      return { interrupted: true }
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
      requestProfile: async (route, method, params) => handle(method, params),
      retainProfile: async () => () => undefined,
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
      '\nglobalThis.__stop = { stopGroupThread, runGroupChatMemberTurn, $groupChats, $groupActivity, currentGroupActivity, updateGroupChat, GROUP_ACTIVITY_LABELS, GROUP_ACTIVITY_GLYPHS };\n'
    )
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  context.plugin.register({
    storage: { get: () => null, set: () => undefined },
    register: () => undefined
  })
  return {
    ...context.__stop,
    context,
    rpcLog,
    calls: method => rpcLog.filter(entry => entry.method === method)
  }
}

const MEMBERS = [
  { name: 'alpha', title: '' },
  { name: 'beta', title: '' },
  { name: 'gamma', title: '' }
]

/** Seed a room mid-round: epoch 3, running, alpha on turn with a live
 *  session id, nobody held yet. */
function seedRoom(gc, { turn = 'alpha' } = {}) {
  gc.$groupChats.set({
    Room: {
      log: [],
      watermarks: {},
      sessions: { alpha: 'live-alpha-sid' },
      epoch: 3,
      running: true,
      turn,
      holds: {},
      members: MEMBERS
    }
  })
}

test('stopGroupThread bumps the epoch, clears running/turn, and holds every member', async () => {
  const gc = load()
  seedRoom(gc)

  await gc.stopGroupThread('Room', 't1', MEMBERS)

  const room = gc.$groupChats.get().Room
  assert.equal(room.epoch, 4, 'epoch bumped — the driving loop bails at its next boundary')
  assert.equal(room.running, false)
  assert.equal(room.turn, null)

  for (const member of MEMBERS) {
    assert.ok(room.holds[member.name], `${member.name} is held — no future turn until an explicit release`)
    assert.equal(room.holds[member.name].thread, 't1')
  }
})

test('stopGroupThread interrupts the member ON TURN via its live session', async () => {
  const gc = load()
  seedRoom(gc, { turn: 'alpha' })

  await gc.stopGroupThread('Room', 't1', MEMBERS)

  const interrupts = gc.calls('session.interrupt')
  assert.equal(interrupts.length, 1, 'exactly one interrupt — the serial loop has one member in flight')
  assert.equal(interrupts[0].params.session_id, 'live-alpha-sid')
})

test('stopGroupThread with nobody on turn stops the room without any interrupt RPC', async () => {
  const gc = load()
  seedRoom(gc, { turn: null })

  await gc.stopGroupThread('Room', 't1', MEMBERS)

  assert.equal(gc.calls('session.interrupt').length, 0)
  assert.equal(gc.$groupChats.get().Room.running, false)
  assert.equal(gc.$groupChats.get().Room.epoch, 4)
})

test('stopGroupThread records a stopped activity event visible in the CURRENT run', async () => {
  const gc = load()
  seedRoom(gc)

  await gc.stopGroupThread('Room', 't1', MEMBERS)

  const events = gc.currentGroupActivity('Room')
  const stopped = events.find(event => event.kind === 'stopped')
  assert.ok(stopped, 'stopped event is tagged with the POST-bump epoch, so it survives the epoch filter')
  assert.equal(stopped.member, 'You')
  assert.equal(stopped.thread, 't1')
  // The label comes from the shared GROUP_ACTIVITY_LABELS map (the plugin's
  // label pattern) — and stays plain English, no hardcoded localized text.
  assert.ok(gc.GROUP_ACTIVITY_LABELS.stopped)
  assert.ok(gc.GROUP_ACTIVITY_GLYPHS.stopped)
})

test('stopGroupThread falls back to the durable room roster when called without members', async () => {
  const gc = load()
  seedRoom(gc)

  await gc.stopGroupThread('Room', 't1')

  const room = gc.$groupChats.get().Room
  assert.equal(Object.keys(room.holds).length, MEMBERS.length)
  assert.equal(gc.calls('session.interrupt').length, 1)
})

test('poll loop abandons an in-flight turn once a stop bumps the epoch and holds the member', async () => {
  let gcRef = null
  const gc = load({
    busyPolls: 50,
    onResumePoll: polls => {
      // Fire the stop from inside the poll cadence, after the second
      // busy poll — exactly the mid-turn click the Stop button produces.
      if (polls === 2) {
        void gcRef.stopGroupThread('Room', 't1', [{ name: 'helper', title: '' }])
      }
    }
  })
  gcRef = gc

  const reply = await gc.runGroupChatMemberTurn('Room', { name: 'helper', title: '' }, 'long task', 't1', [])

  assert.equal(reply, null, 'abandoned turn yields no reply')
  const postStopPolls = gc.calls('session.resume').length
  assert.ok(postStopPolls <= 6, `poll loop exited promptly after the stop, not at the deadline (${postStopPolls} resumes)`)
  assert.equal(gc.$groupChats.get().Room.running, false)
})

test('an ordinary newer-send epoch bump WITHOUT a hold does not abandon the poll — late work still lands', async () => {
  let gcRef = null
  const gc = load({
    reply: 'finished anyway',
    busyPolls: 3,
    onResumePoll: polls => {
      if (polls === 1) {
        // A newer user send bumps the epoch but holds nobody. The in-flight
        // poll must keep going so the finished reply can still be delivered
        // (the #93127 commit check decides its fate at the boundary).
        gcRef.updateGroupChat('Room', r => {
          r.epoch = (r.epoch || 0) + 1
          return r
        })
      }
    }
  })
  gcRef = gc

  const reply = await gc.runGroupChatMemberTurn('Room', { name: 'helper', title: '' }, 'long task', 't1', [])

  assert.equal(reply, 'finished anyway', 'epoch churn alone never abandons a live turn')
})

