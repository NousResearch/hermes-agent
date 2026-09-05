import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as data from './data'
import type * as groupChat from './group-chat'
import type * as groupMembership from './group-membership'
import { createGroupGateway, scriptedStorage } from './group-test-utils'
import type * as labels from './labels'
import type { BotMeta, GroupChat, GroupMember, RosterRow } from './types'

// Who is seated in a room, and how a room's membership is written down.
// Membership lives in two places on purpose: local bots carry it in bot-meta
// (`groups`, syncable via ui_meta), while remote members can only be described
// by the room record itself — so seating is a union, and the two halves have
// to agree about identity.

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

interface Modules {
  chat: typeof groupChat
  data: typeof data
  labels: typeof labels
  membership: typeof groupMembership
}

async function load(): Promise<Modules> {
  vi.resetModules()
  const gateway = createGroupGateway()

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, gateway.host)

  const [chat, data, labels, membership, shared] = await Promise.all([
    import('./group-chat'),
    import('./data'),
    import('./labels'),
    import('./group-membership'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))

  return { chat, data, labels, membership }
}

let modules: Modules

/** Room records as the store holds them, built from the handful of fields a
 *  case actually cares about. Runtime `turn` included — presence tests key
 *  off it. */
function rooms(map: Record<string, Partial<groupChat.GroupChatRoom>>): Record<string, groupChat.GroupChatRoom> {
  return Object.fromEntries(Object.entries(map).map(([name, room]) => [name, { log: [], watermarks: {}, ...room }]))
}

beforeEach(async () => {
  modules = await load()
})

describe('membership metadata', () => {
  it('botGroups normalizes canonical and legacy membership without duplicates', () => {
    const { botGroups } = modules.membership

    expect(
      botGroups({
        group: 'Operations',
        groups: [' Engineering ', '', 'Research', 'Engineering', null, 7, { name: 'Nope' }]
      } as BotMeta)
    ).toEqual(['Engineering', 'Research'])
    expect(botGroups({ group: 'Legacy' })).toEqual(['Legacy'])
    expect(botGroups({ groups: [] })).toEqual([])
  })

  it('groupMembershipPatch toggles one membership and keeps the legacy projection compatible', () => {
    const { groupMembershipPatch } = modules.membership
    const meta: BotMeta = { group: 'Engineering', groups: ['Engineering', 'Research'] }

    expect(groupMembershipPatch(meta, 'Engineering', true)).toEqual({
      group: 'Engineering',
      groups: ['Engineering', 'Research']
    })
    expect(groupMembershipPatch(meta, 'Operations', true)).toEqual({
      group: 'Engineering',
      groups: ['Engineering', 'Research', 'Operations']
    })
    expect(groupMembershipPatch(meta, 'Engineering', false)).toEqual({ group: 'Research', groups: ['Research'] })
    expect(groupMembershipPatch({ group: 'Legacy' }, 'Legacy', false)).toEqual({ group: null, groups: [] })
  })

  it('knownGroups is unique, trimmed and alphabetical', () => {
    expect(
      modules.membership.knownGroups({
        a: { group: 'research' },
        b: { group: 'Ops', groups: ['Ops', 'research'] },
        c: { groups: ['Design'] },
        d: { group: '' },
        e: {}
      })
    ).toEqual(['Design', 'Ops', 'research'])
  })
})

describe('room listing', () => {
  it('groupChatNames unions bot-meta groups with room records that carry members or a log', () => {
    const meta: Record<string, BotMeta> = {
      pm: { group: 'Ops', groups: ['Ops', 'Research'] },
      researcher: { group: 'Research' },
      scout: { group: 'Stale', groups: ['External'] }
    }

    const known = rooms({
      Chatty: { log: [{ at: 5, from: { kind: 'user', name: 'You' }, id: '1', text: 'hi' }] },
      Empty: { members: [] }, // nothing behind it — no row
      Remote: { members: [{ name: 'spark', remoteSource: true }] },
      Research: { members: [] } // already known via meta
    })

    expect([...modules.membership.groupChatNames(meta, known)].sort()).toEqual([
      'Chatty',
      'External',
      'Ops',
      'Remote',
      'Research'
    ])
  })

  it('groupLastActivity is the newest room-log timestamp, 0 for silence', () => {
    const { groupLastActivity } = modules.membership

    expect(groupLastActivity({ log: [{ at: 3 }, { at: 9 }] } as GroupChat)).toBe(9)
    expect(groupLastActivity({ log: [], watermarks: {} })).toBe(0)
    expect(groupLastActivity(undefined)).toBe(0)
  })

  it('stripPreviewMarkdown flattens bold, quotes, code and links out of row previews', () => {
    const { stripPreviewMarkdown } = modules.labels

    expect(stripPreviewMarkdown('**Plan**: ship the `thing`')).toBe('Plan: ship the thing')
    expect(stripPreviewMarkdown('> quoted wisdom')).toBe('quoted wisdom')
    expect(stripPreviewMarkdown('see [the doc](https://x.y/z) now')).toBe('see the doc now')
    expect(stripPreviewMarkdown('## Heading\nbody')).toBe('Heading body')
    expect(stripPreviewMarkdown('')).toBe('')
  })
})

describe('seating a room', () => {
  it('seats local meta members plus stored remote descriptors, preferring live rows', () => {
    const roster: RosterRow[] = [
      { name: 'researcher' },
      { name: 'builder' },
      { connectionId: 'c1', name: 'spark', remoteSource: true, sourceScoped: true }
    ]

    modules.chat.$groupChats.set(
      rooms({
        Research: {
          log: [],
          members: [{ connectionId: 'c1', name: 'spark', remoteSource: true, sourceScoped: true }]
        }
      })
    )

    const members = modules.membership.groupChatMemberBots('Research', roster, {
      builder: { group: 'Ops', groups: ['Ops', 'Research'] },
      researcher: { group: 'Research' }
    })

    expect(members.map(row => row.name)).toEqual(['researcher', 'builder', 'spark'])
    // The LIVE roster row was preferred over the stored descriptor.
    expect(members[2]).toBe(roster[2])
  })

  it('keeps a persisted unreachable source member seated next to a live same-name twin', () => {
    const live: RosterRow = {
      connectionId: 'local',
      connectionKind: 'local',
      connectionLabel: 'This device',
      handle: 'profile-a-this-device',
      name: 'profile-a',
      sourceReachable: true
    }

    const stored: RosterRow = {
      connectionId: 'loopback-19119',
      connectionKind: 'remote',
      connectionLabel: '127.0.0.1:19119',
      handle: 'profile-a-127-0-0-1-19119',
      name: 'profile-a',
      remoteSource: true,
      sourceReachable: false,
      sourceScoped: true
    }

    modules.chat.$groupChats.set(rooms({ Research: { log: [], members: [stored] } }))

    const members = modules.membership.groupChatMemberBots('Research', [live, stored], {
      'profile-a': { group: 'Research' }
    })

    expect(members).toHaveLength(2)
    expect(members[0]).toBe(live)
    expect(members[1]).toBe(stored)
    expect(members[1].handle).toBe('profile-a-127-0-0-1-19119')
  })

  it('lets a stored descriptor beat a presentation-only ghost', () => {
    // A selected-but-offline ghost carries only enough identity to paint the
    // roster; the durable descriptor owns the handle mentions resolve against.
    const ghost: RosterRow = {
      connectionId: 'c1',
      connectionLabel: 'Workshop',
      ghost: true,
      name: 'spark',
      remoteSource: true,
      sourceScoped: true
    }

    const stored: RosterRow = {
      connectionId: 'c1',
      connectionLabel: 'Workshop',
      handle: 'spark-work',
      name: 'spark',
      remoteSource: true,
      sourceScoped: true,
      title: 'Spark'
    }

    modules.chat.$groupChats.set(rooms({ Research: { log: [], members: [stored] } }))

    const members = modules.membership.groupChatMemberBots('Research', [ghost], {})

    expect(members).toHaveLength(1)
    expect(members[0]).toBe(stored)
    expect(members[0].handle).toBe('spark-work')
  })

  it('durableGroupChatMembers retains active and remote source identities', () => {
    const members = modules.membership.durableGroupChatMembers([
      {
        connectionId: 'noah',
        connectionKind: 'remote',
        connectionLabel: 'Noah',
        handle: 'noah',
        name: 'default',
        sourceScoped: true
      },
      {
        connectionId: 'maya',
        connectionKind: 'remote',
        connectionLabel: 'Maya',
        handle: 'maya',
        name: 'default',
        remoteSource: true,
        sourceScoped: true
      }
    ])

    expect(members).toEqual([
      {
        connectionId: 'noah',
        connectionKind: 'remote',
        connectionLabel: 'Noah',
        handle: 'noah',
        name: 'default',
        remoteSource: true,
        route: { connectionId: 'noah', mode: 'remote', profile: 'default', targetProfile: 'default' },
        sourceScoped: true,
        targetProfile: 'default'
      },
      {
        connectionId: 'maya',
        connectionKind: 'remote',
        connectionLabel: 'Maya',
        handle: 'maya',
        name: 'default',
        remoteSource: true,
        route: { connectionId: 'maya', mode: 'remote', profile: 'default', targetProfile: 'default' },
        sourceScoped: true,
        targetProfile: 'default'
      }
    ])
  })
})

// #92794: older builds persisted group members with a FRIENDLY name as `name`
// (e.g. `name: '大司命'` for the profile slug `taiyi`). Key matching alone
// seats such a descriptor as a ghost NEXT TO its own live roster row ("4 bots"
// in a 2-bot room — reproduced live), and any path that passes the ghost's
// identity onward targets a profile that does not exist on disk. Seating
// re-tries an unmatched descriptor by friendly name against same-connection
// roster rows first.
describe('legacy display-name descriptors', () => {
  const TAIYI: RosterRow = { connectionId: 'local', display_name: '大司命', name: 'taiyi', remoteSource: false }
  const TESTBOT: RosterRow = { connectionId: 'local', display_name: '', name: 'testbot', remoteSource: false }

  it('seat their live row once, not as extra ghosts', () => {
    modules.chat.$groupChats.set(
      rooms({
        room: {
          log: [],
          // The legacy shape: friendly names persisted as `name`.
          members: [
            { connectionId: 'local', handle: '大司命', name: '大司命' },
            { connectionId: 'local', handle: 'Testbot', name: 'Testbot' }
          ]
        }
      })
    )
    modules.data.$botMeta.set({ taiyi: { groups: ['room'] }, testbot: { groups: ['room'], title: 'Testbot' } })

    const seated = modules.membership.groupChatMemberBots('room', [TAIYI, TESTBOT], {
      taiyi: { groups: ['room'] },
      testbot: { groups: ['room'], title: 'Testbot' }
    })

    // Two members, not four: each legacy descriptor resolved to its live row.
    expect(seated.map(row => row.name).sort()).toEqual(['taiyi', 'testbot'])
  })

  it('still seat a genuinely unknown descriptor as a degraded ghost', () => {
    modules.chat.$groupChats.set(rooms({ room: { log: [], members: [{ connectionId: 'gone', name: 'vanished' }] } }))

    const seated = modules.membership.groupChatMemberBots('room', [TAIYI], { taiyi: { groups: ['room'] } })

    expect(seated).toHaveLength(2)
    expect(seated.some(row => row.name === 'vanished')).toBe(true)
  })

  it('resolve a connectionless pre-scoping descriptor against local rows', () => {
    // The oldest legacy shape: no connectionId at all (pre-connection-scoping
    // rooms only ever held this machine's bots). Reproduced live: such ghosts
    // keyed as legacy::<display-name> and doubled the seated roster.
    modules.chat.$groupChats.set(rooms({ room: { log: [], members: [{ name: '大司命' }] } }))

    expect(modules.membership.groupChatMemberBots('room', [TAIYI], {}).map(row => row.name)).toEqual(['taiyi'])
  })

  it('never match a friendly name across connections', () => {
    const remoteTwin: RosterRow = {
      connectionId: 'other-box',
      display_name: '大司命',
      name: 'shadow',
      remoteSource: true
    }

    modules.chat.$groupChats.set(rooms({ room: { log: [], members: [{ connectionId: 'local', name: '大司命' }] } }))

    const seated = modules.membership.groupChatMemberBots('room', [remoteTwin], {})

    // Same friendly name on ANOTHER connection must not capture the member.
    expect(seated.map(row => row.name)).toEqual(['大司命'])
    expect(seated[0].connectionId).toBe('local')
  })

  it('pass an exact slug descriptor through untouched', () => {
    const descriptor = {
      connectionId: 'local',
      name: 'taiyi',
      route: { connectionId: 'local', mode: 'local', profile: 'taiyi', targetProfile: 'taiyi' }
    } as GroupMember

    modules.chat.$groupChats.set(rooms({ room: { log: [], members: [descriptor] } }))

    expect(modules.membership.groupChatMemberBots('room', [TAIYI], {})[0]).toBe(TAIYI)
  })
})

// Presence: a member mid group turn lights its roster row (and, through the
// member keys, the group row). The room's runtime `turn` is the same identity
// the thread's thinking line uses, so the roster cannot disagree with the room.
describe('activeGroupMemberKeys', () => {
  const RESEARCHER: RosterRow = { connectionId: 'local', name: 'researcher', remoteSource: false }
  const SCRIBE: RosterRow = { connectionId: 'local', name: 'scribe', remoteSource: false }

  it('maps the room turn to the live source-qualified roster key', () => {
    modules.chat.$groupChats.set(rooms({ room: { log: [], running: true, turn: 'researcher' } }))
    modules.data.$botMeta.set({ researcher: { groups: ['room'] } })

    const keys = modules.membership.activeGroupMemberKeys(
      modules.chat.$groupChats.get(),
      [RESEARCHER, SCRIBE],
      { researcher: { groups: ['room'] } }
    )

    expect(keys).toEqual(['local::researcher'])
  })

  it('seats a remote member through its stored descriptor and qualifies the key', () => {
    const dixie: RosterRow = { connectionId: 'mini', name: 'dixie', remoteSource: true }

    modules.chat.$groupChats.set(
      rooms({ room: { log: [], members: [{ connectionId: 'mini', name: 'dixie' }], running: true, turn: 'dixie' } })
    )

    expect(modules.membership.activeGroupMemberKeys(modules.chat.$groupChats.get(), [dixie], {})).toEqual(['mini::dixie'])
  })

  it('stays empty when the room is idle, nobody is on turn, or the member is unseated', () => {
    modules.chat.$groupChats.set(rooms({ room: { log: [], running: true } }))

    expect(modules.membership.activeGroupMemberKeys(modules.chat.$groupChats.get(), [RESEARCHER], {})).toEqual([])

    modules.chat.$groupChats.set(rooms({ room: { log: [], running: true, turn: 'ghost' } }))

    expect(modules.membership.activeGroupMemberKeys(modules.chat.$groupChats.get(), [RESEARCHER], {})).toEqual([])

    modules.chat.$groupChats.set(rooms({ room: { log: [], running: false, turn: 'researcher' } }))

    expect(modules.membership.activeGroupMemberKeys(modules.chat.$groupChats.get(), [RESEARCHER], {})).toEqual([])
  })
})
