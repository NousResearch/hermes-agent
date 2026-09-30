/**
 * #129443 — the redundancy "(pass)" must not swallow an explicit address.
 *
 * The room's participation rules tell a member to reply only with something
 * new and otherwise "(pass)" — so in an @everyone turn the second member
 * passes whenever the first already answered, and a directly addressed
 * member's silence settles as ordinary consensus. The engine, not the prompt
 * wording, owns the invariant: `explicitlyAddressedMemberKeys` carries the
 * user's addresses as structured state, an addressed member's "(pass)" gets
 * ONE bounded nudge, and a second "(pass)" is recorded as an explicit
 * noncompliance (activity row + roster badge) instead of quiet silence.
 * Ordinary unaddressed collaborative "(pass)" is untouched.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as groupActivity from './group-activity'
import type * as groupChat from './group-chat'
import type * as groupMembership from './group-membership'
import type * as groupRounds from './group-rounds'
import { createGroupGateway, drain, runTimersInline, scriptedStorage } from './group-test-utils'
import type { GatewayOptions, ScriptedGateway } from './group-test-utils'
import type { GroupMember, GroupMessage } from './types'

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

interface Room {
  activity: typeof groupActivity
  chat: typeof groupChat
  gateway: ScriptedGateway
  membership: typeof groupMembership
  rounds: typeof groupRounds
}

async function loadRoom(options: GatewayOptions = {}): Promise<Room> {
  vi.resetModules()
  const gateway = createGroupGateway(options)

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, gateway.host)

  const [activity, chat, membership, rounds, shared] = await Promise.all([
    import('./group-activity'),
    import('./group-chat'),
    import('./group-membership'),
    import('./group-rounds'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))

  return { activity, chat, gateway, rounds, membership }
}

const MEMBERS: GroupMember[] = [
  { name: 'research', title: '' },
  { name: 'builder', title: '' }
]

const log = (room: Room, group: string) => room.chat.$groupChats.get()[group]?.log || []

/** Run the room's drive to completion. */
async function settle(room: Room, group: string) {
  await drain(() => Boolean(room.chat.$groupChats.get()[group]?.running))
}

beforeEach(() => {
  runTimersInline()
})

describe('explicitlyAddressedMemberKeys', () => {
  const user = (text: string): GroupMessage[] =>
    [{ at: 1, from: { kind: 'user', name: 'You' }, text }] as GroupMessage[]

  it('addresses every member on @everyone, only the named one on a bare mention, nobody without a mention', async () => {
    const { rounds, membership } = await loadRoom()
    const keysOf = (members: GroupMember[]) => members.map(member => membership.groupMemberKey(member))

    expect([...rounds.explicitlyAddressedMemberKeys(user('@everyone standup'), MEMBERS)].sort()).toEqual(
      [...keysOf(MEMBERS)].sort()
    )
    expect([...rounds.explicitlyAddressedMemberKeys(user('@builder take this one'), MEMBERS)]).toEqual([
      membership.groupMemberKey(MEMBERS[1])
    ])
    expect(rounds.explicitlyAddressedMemberKeys(user('fyi, deploy went out'), MEMBERS).size).toBe(0)
  })

  it('never counts a member reply\'s @handoff as a user address', async () => {
    const { rounds, membership } = await loadRoom()

    const entries: GroupMessage[] = [
      ...user('@builder take this one'),
      { at: 2, from: { kind: 'member', name: 'builder' }, text: 'on it — @research can you dig in?' } as GroupMessage
    ]

    expect([...rounds.explicitlyAddressedMemberKeys(entries, MEMBERS)]).toEqual([
      membership.groupMemberKey(MEMBERS[1])
    ])
  })
})

describe('an addressed member that (pass)ed', () => {
  it('gets one nudge and its recovered reply is posted to the room', async () => {
    const room = await loadRoom({
      turn: ({ prompt }) => (prompt.includes('explicitly addressed') ? 'here is my answer' : '(pass)')
    })

    room.rounds.sendToGroupChat('Nudge', MEMBERS, '@builder status?')
    await settle(room, 'Nudge')

    const builderCalls = room.gateway.calls.filter(call => call.profile === 'builder')

    expect(builderCalls).toHaveLength(2)
    expect(builderCalls[0].prompt).not.toContain('explicitly addressed')
    expect(builderCalls[1].prompt).toContain('explicitly addressed')
    expect(
      log(room, 'Nudge')
        .filter(entry => entry.from.kind === 'member')
        .map(entry => entry.text)
    ).toEqual(['here is my answer'])
    expect(room.activity.currentGroupActivity('Nudge').filter(event => event.kind === 'failed')).toEqual([])
  })

  it('records explicit noncompliance (never plain silence) when the nudge is passed on too', async () => {
    const room = await loadRoom({ turn: () => '(pass)' })

    room.rounds.sendToGroupChat('Stubborn', MEMBERS, '@builder status?')
    await settle(room, 'Stubborn')

    const builderCalls = room.gateway.calls.filter(call => call.profile === 'builder')

    expect(builderCalls).toHaveLength(2)
    expect(log(room, 'Stubborn').filter(entry => entry.from.kind === 'member')).toEqual([])
    expect(room.activity.currentGroupActivity('Stubborn').filter(event => event.kind === 'failed')).toEqual([
      expect.objectContaining({
        kind: 'failed',
        member: room.membership.groupMemberKey(MEMBERS[1]),
        reason: 'explicitly addressed member passed twice'
      })
    ])
  })

  it('leaves an ordinary collaborative pass alone (no mention, no nudge, no failure)', async () => {
    const room = await loadRoom({ turn: () => '(pass)' })

    room.rounds.sendToGroupChat('Quiet', MEMBERS, 'fyi, deploy went out')
    await settle(room, 'Quiet')

    expect(room.gateway.calls).toHaveLength(2)
    expect(room.gateway.calls.every(call => !call.prompt.includes('explicitly addressed'))).toBe(true)
    expect(log(room, 'Quiet')).toHaveLength(1)
    expect(log(room, 'Quiet')[0].from.kind).toBe('user')
    expect(room.activity.currentGroupActivity('Quiet').filter(event => event.kind === 'failed')).toEqual([])
  })
})
