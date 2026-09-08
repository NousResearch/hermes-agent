import { existsSync, readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as groupActivity from './group-activity'
import type * as groupChat from './group-chat'
import { groupMemberKey } from './group-membership'
import type * as groupRounds from './group-rounds'
import { createGroupGateway, drain, runTimersInline, scriptedStorage } from './group-test-utils'
import type { GatewayOptions, ScriptedGateway } from './group-test-utils'
import type * as groupTurns from './group-turns'
import type { GroupMember } from './types'

// The same scripted rooms the hosted-room policy runs in tests/gateway/test_hosted_room_rounds.py.
// Both drivers answer them with their real round loop, so "who speaks next" and "what a resumed
// member sees" cannot come to mean two different things depending on which frontend was used.
// Whole traces on purpose: equal caps and equal rotation helpers do not make equal conversations.

const { host, observed } = vi.hoisted(() => ({
  host: {} as Record<string, unknown>,
  /** Ordered runtime observations: 'harvest' opens each ordinary round, 'turn' is one dispatch. */
  observed: [] as Array<{ kind: 'harvest' | 'turn'; member: string; ordinary: boolean }>
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

// Read-only wrappers over the REAL turn module. Nothing is replaced: each call is recorded and
// delegated. The two runtime facts they expose are the drive's own, not an inference — every
// ordinary round opens by harvesting each member, and only the ordinary dispatch passes the
// round's delta images (group-rounds.ts:646 against its recovery call at :813).
vi.mock('./group-turns', async () => {
  const actual = await vi.importActual<typeof groupTurns>('./group-turns')

  return {
    ...actual,
    harvestStrandedGroupReply: (...args: Parameters<typeof actual.harvestStrandedGroupReply>) => {
      observed.push({
        kind: 'harvest', member: (args[1] as { name: string })?.name, ordinary: true
      })

      return actual.harvestStrandedGroupReply(...args)
    },
    runGroupChatMemberTurn: (...args: Parameters<typeof actual.runGroupChatMemberTurn>) => {
      observed.push({
        kind: 'turn', member: (args[1] as { name: string })?.name, ordinary: args.length >= 5
      })

      return actual.runGroupChatMemberTurn(...args)
    }
  }
})

function repoFile(relative: string): string {
  for (let directory = resolve(process.cwd()); ; directory = dirname(directory)) {
    const candidate = join(directory, relative)

    if (existsSync(candidate)) {return candidate}

    if (dirname(directory) === directory) {throw new Error(`${relative} was not found above ${process.cwd()}`)}
  }
}

interface Scenario {
  name: string
  group: string
  members: string[]
  user: string[]
  /** Per member: one flat list of replies, or a list per user message (keyed by its 1-based
   *  index) when a scenario needs a member's answer to belong to a specific discussion. */
  replies: Record<string, Record<string, string[]> | string[]>
  speakers: string[]
  /** Every member the drive actually asked for a turn, in submission order — passes included,
   *  because a silent recovery turn leaves no trace in the log at all. */
  dispatch?: string[]
  /** The drive's own terminal activity: `settled` or `capped`. */
  outcome?: string
  /** Per dispatch: `[member, 'ordinary' | 'continuation', round]`, as the drive itself ran it. */
  phases?: Array<[string, string, number]>
  /** What a held member has and has not read: the last entry its skips consumed, and a later
   *  entry that must still be waiting for it. */
  unread?: { member: string; read: string; waiting: string }
  resumed_delta?: { member: string; must_contain: string[] }
}

const vectors = JSON.parse(
  readFileSync(repoFile('tests/fixtures/hosted_room_round_traces.json'), 'utf8')
) as { members: Array<{ member_id: string; handle: string }>; scenarios: Scenario[] }

interface Room {
  activity: typeof groupActivity
  chat: typeof groupChat
  gateway: ScriptedGateway
  rounds: typeof groupRounds
}

async function loadRoom(options: GatewayOptions = {}): Promise<Room> {
  vi.resetModules()
  const gateway = createGroupGateway(options)

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, gateway.host)

  const [activity, chat, rounds, shared] = await Promise.all([
    import('./group-activity'),
    import('./group-chat'),
    import('./group-rounds'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))

  return { activity, chat, gateway, rounds }
}

/** Drive one scenario through the real round loop, awaiting each user message's own drive.
 *  The first send mints the thread; every later message continues THAT thread, because an
 *  omitted thread argument deliberately starts a new one and a release has to reach the same
 *  conversation the hold was set in. */
/** Per dispatch: the member, the round its drive was in, and whether it was an ordinary slot. */
function phasesFrom(entries: typeof observed): Array<[string, string, number]> {
  const phases: Array<[string, string, number]> = []
  let round = -1
  let opening = false

  for (const entry of entries) {
    if (entry.kind === 'harvest') {
      if (!opening) {round += 1}

      opening = true

      continue
    }

    if (round < 0) {
      // Every ordinary round opens by harvesting each member. A dispatch before any harvest
      // means this seam no longer observes what it claims to; say so instead of reporting a
      // round zero that was never seen.
      throw new Error(`dispatch to ${entry.member} was not preceded by a round harvest`)
    }

    opening = false
    phases.push([entry.member, entry.ordinary ? 'ordinary' : 'continuation', round])
  }

  return phases
}

async function runScenario(scenario: Scenario) {
  observed.length = 0
  const spoken = new Map<string, number>()
  /** Every scripted turn in submission order: who was asked, and what it answered. */
  const turns: Array<{ profile: string; reply: string }> = []
  let phase = 1

  /** A flat list is ONE continuous script for the whole scenario, exactly as the accepted
   *  vectors use it. Only the per-discussion map restarts with each user message. */
  const scriptFor = (profile: string): { key: string; lines: string[] } => {
    const scripted = scenario.replies[profile]

    return Array.isArray(scripted)
      ? { key: profile, lines: scripted }
      : { key: `${profile}:${phase}`, lines: scripted?.[String(phase)] || [] }
  }

  const room = await loadRoom({
    turn: ({ profile }) => {
      const { key, lines } = scriptFor(profile)
      const index = spoken.get(key) || 0
      spoken.set(key, index + 1)
      const reply = lines[index] ?? '(pass)'
      turns.push({ profile, reply })

      return reply
    }
  })

  const members: GroupMember[] = scenario.members.map(handle => ({ name: handle, title: '' }))
  let thread: null | string = null
  const exhausted: string[] = []
  // Each send starts its own drive, whose rounds count from zero again.
  const segments: number[] = []

  for (const [index, text] of scenario.user.entries()) {
    // Sequential sends into the SAME thread: the discussion index a per-discussion script keys on.
    phase = index + 1
    segments.push(observed.length)
    const minted = room.rounds.sendToGroupChat(scenario.group, members, text, thread)
    thread = thread || minted || null
    await drain(() => Boolean(room.chat.$groupChats.get()[scenario.group]?.running))

    // `drain` gives up after a bounded number of turns of the event loop. A drive still
    // running here means the trace was cut short, which must never read as a settled room.
    if (room.chat.$groupChats.get()[scenario.group]?.running) {
      exhausted.push(text)
    }
  }

  const stored = room.chat.$groupChats.get()[scenario.group] || { log: [], watermarks: {} }
  const log = stored.log || []

  return {
    // The room's own stored state: the held skip writes these marks, so what a paused member
    // has and has not read is observable here rather than only in the other engine.
    watermarks: stored.watermarks || {},
    texts: log.map(entry => entry.text),
    thread,
    speakers: log.filter(entry => entry.from.kind === 'member').map(entry => entry.from.name),
    threads: [...new Set(log.map(entry => entry.thread))],
    exhausted,
    phases: segments.flatMap((start, index) =>
      phasesFrom(observed.slice(start, segments[index + 1] ?? observed.length))),
    // Diagnostics: what the engine actually did, so a mismatch is explained by evidence
    // instead of a guess about rounds, rotation or holds.
    diagnostics: {
      turns,
      committed: log.map(entry => `${entry.from.kind}:${entry.from.name}: ${entry.text}`),
      deltas: room.gateway.calls.map(call => ({ profile: call.profile, prompt: call.prompt })),
      activity: (room.activity.$groupActivity.get()[scenario.group]?.events || []).map(
        event => `${event.kind}:${event.member ?? '-'}`
      )
    }
  }
}

beforeEach(() => {
  runTimersInline()
  // Entries are timestamped with Date.now (group-chat.ts) and the sync merge orders equal-time
  // entries by generated key, so inline timers can persist a round in an order the drive never
  // dispatched. A strictly increasing clock removes that tie-break from the policy comparison;
  // the real same-millisecond ordering is a separate persistence question, not this contract.
  let tick = Date.now()
  vi.spyOn(Date, 'now').mockImplementation(() => ++tick)
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('shared round traces', () => {
  it.each(vectors.scenarios.map(one => [one.name, one] as const))('%s', async (_name, scenario) => {
    const actual = await runScenario(scenario)
    const evidence = JSON.stringify(actual.diagnostics, null, 1)

    // A drive that was still running when the harness stopped draining did not settle; its
    // trace is truncated, not a contract.
    expect(actual.exhausted, evidence).toEqual([])
    expect(actual.speakers, evidence).toEqual(scenario.speakers)

    const dispatch = actual.diagnostics.turns.map(turn => turn.profile)

    const outcome = actual.diagnostics.activity
      .map(entry => entry.split(':')[0])
      .filter(kind => ['cancelled', 'capped', 'settled'].includes(kind))
      .at(-1)

    if (scenario.dispatch || scenario.outcome) {
      // Who was ASKED, not only who spoke, and how the drive ended.
      expect({ dispatch, outcome }, evidence).toEqual({
        dispatch: scenario.dispatch ?? dispatch, outcome: scenario.outcome ?? outcome
      })
    }

    if (scenario.phases) {
      // Round and ordinary/recovery phase per dispatch, read from the drive's own calls.
      expect(actual.phases, evidence).toEqual(scenario.phases.map(entry => [...entry]))
    }

    // One conversation: a scenario that silently forked into a second thread would compare
    // unrelated deltas on both sides.
    expect(actual.threads).toHaveLength(1)

    if (scenario.unread) {
      const key = `${actual.thread}::${groupMemberKey({ name: scenario.unread.member, title: '' })}`
      const mark = Number((actual.watermarks as Record<string, number>)[key] ?? 0)
      const read = actual.texts.indexOf(scenario.unread.read)
      const waiting = actual.texts.indexOf(scenario.unread.waiting)

      expect([read, waiting].every(index => index >= 0), evidence).toBe(true)
      // Its skips consumed the earlier entry; the later one is still ahead of its mark.
      expect(mark, evidence).toBeGreaterThan(read)
      expect(mark, evidence).toBeLessThanOrEqual(waiting)
    }

    if (scenario.resumed_delta) {
      const delivered = actual.diagnostics.deltas.filter(
        call => call.profile === scenario.resumed_delta!.member)

      const last = delivered.at(-1)?.prompt || ''

      for (const fragment of scenario.resumed_delta.must_contain) {
        expect(last).toContain(fragment)
      }
    }
  })
})
