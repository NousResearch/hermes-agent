/**
 * crash ≠ consensus — the degraded-vs-settled split (M3 gate item).
 *
 * The round loop historically treated pass / nothing / FAILED alike as
 * silence, so a settled round could not distinguish chosen-quiet from a
 * crashed seat: a room where a member died mid-drive attested "everyone
 * passed" while a voice was actually missing (its own exit comment admitted
 * it; feed kinds 'failed'/'timed-out' existed but the settle decision ignored
 * them). The drive must require POSITIVE evidence — a real pass — before it
 * may call the quiet consensus. A member turn ending in failure this drive
 * makes the quiet exit DEGRADED instead of SETTLED, with the failed-seat
 * count surfaced in the activity feed (why 770: "this round could not
 * settle — N seats failed", visible in the room, not via ledger archaeology).
 *
 * Deliberately NOT degraded: held seats (user-chosen quiet), timed-out turns
 * (they surface their own row and a late reply can still land), and an
 * addressed member that passed twice (a positive refusal is evidence, a hole
 * in the evidence is not).
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as groupActivity from './group-activity'
import type * as groupChat from './group-chat'
import type * as groupRounds from './group-rounds'
import { createGroupGateway, drain, runTimersInline, scriptedStorage } from './group-test-utils'
import type { GatewayOptions, ScriptedGateway } from './group-test-utils'
import type { GroupMember } from './types'

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

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

const MEMBERS: GroupMember[] = [
  { name: 'research', title: '' },
  { name: 'builder', title: '' },
  { name: 'ops', title: 'The Ops' }
]

/** Run the room's drive to completion. */
async function settle(room: Room, group: string) {
  await drain(() => Boolean(room.chat.$groupChats.get()[group]?.running))
}

const kinds = (room: Room, group: string) =>
  room.activity.currentGroupActivity(group).map(event => event.kind)

const exitRow = (room: Room, group: string) => room.activity.currentGroupActivity(group).at(-1)!

beforeEach(() => {
  runTimersInline()
})

describe('degraded vs settled', () => {
  it('a crashed seat during an otherwise-quiet drive ends DEGRADED, not settled', async () => {
    const room = await loadRoom({
      turn: ({ profile }) => {
        if (profile === 'builder') {
          throw new Error('gateway hiccup')
        }

        return '(pass)'
      }
    })

    room.rounds.sendToGroupChat('Crash', MEMBERS, 'anyone around?')
    await settle(room, 'Crash')

    const events = room.activity.currentGroupActivity('Crash')

    expect(kinds(room, 'Crash').at(-1)).toBe('degraded')
    expect(kinds(room, 'Crash')).not.toContain('settled')

    // The exit row names how many voices are missing from the quiet.
    expect(exitRow(room, 'Crash').reason).toBe('1 seat failed')

    // The failure itself stays visible with its cause, like before.
    expect(events.some(event => event.kind === 'failed' && event.member === 'builder')).toBe(true)
  })

  it('all-pass with zero failures still settles as consensus', async () => {
    const room = await loadRoom({ turn: () => '(pass)' })

    room.rounds.sendToGroupChat('Quiet', MEMBERS, 'standup')
    await settle(room, 'Quiet')

    expect(kinds(room, 'Quiet').at(-1)).toBe('settled')
    expect(kinds(room, 'Quiet')).not.toContain('degraded')
  })

  it('degrades on an inherited failure the drive never re-asked (#93127 short-circuit)', async () => {
    // A failure seeded from an earlier queued thread: the short-circuit
    // skips that member WITHOUT recording a fresh failed row this drive,
    // and its silence must still block consensus — crash-shaped absence
    // is not a pass.
    const room = await loadRoom({ turn: () => '(pass)' })

    // Warm the room record with a clean consensus drive.
    room.rounds.sendToGroupChat('Inherit', MEMBERS, 'warm up')
    await settle(room, 'Inherit')
    expect(kinds(room, 'Inherit').at(-1)).toBe('settled')

    room.chat.appendGroupChatEntry('Inherit', { kind: 'user', name: 'You' }, 'go ahead', 't2')
    await room.rounds.runGroupChatRounds('Inherit', MEMBERS, 't2', new Set(['builder']))

    expect(kinds(room, 'Inherit').at(-1)).toBe('degraded')
    expect(exitRow(room, 'Inherit').reason).toBe('1 seat failed')
    // No fresh failed row this drive: the skip is silent at member level,
    // and the degraded exit is what surfaces it.
    expect(kinds(room, 'Inherit').slice(1)).not.toContain('failed')
  })

  it('a spoken reply after a crash still ends degraded, never settled', async () => {
    // Round 1: research replies (someone spoke), builder's turn crashes.
    // Round 2: research has no new delta and builder crashes again — the
    // round goes quiet with a crashed seat on record. The quiet consensus
    // check at spokeThisRound === 0 must see failedMembers and refuse to
    // call this consensus.
    const room = await loadRoom({
      turn: ({ profile }) => {
        if (profile === 'builder') {
          throw new Error('backend exploded')
        }

        return 'ship it.'
      }
    })

    const two: GroupMember[] = [
      { name: 'research', title: '' },
      { name: 'builder', title: '' }
    ]

    room.rounds.sendToGroupChat('Mixed', two, 'anyone around?')
    await settle(room, 'Mixed')

    expect(kinds(room, 'Mixed')).toContain('replied')
    expect(kinds(room, 'Mixed').at(-1)).toBe('degraded')
  })

  it('label renders the missing-voice count for the room feed', async () => {
    const room = await loadRoom({ turn: () => '(pass)' })

    // The recorder needs a room record to stamp the epoch onto.
    room.chat.appendGroupChatEntry('Label', { kind: 'user', name: 'You' }, 'hello', 't1')

    const entry = room.activity.recordGroupActivity('Label', {
      kind: 'degraded',
      member: null,
      reason: '2 seats failed',
      thread: 't1'
    })

    expect(room.activity.groupActivityLabel(entry!)).toBe('the round could not settle — 2 seats failed')
    expect(room.activity.groupActivityTone('degraded')).toBe('text-destructive')
  })
})
